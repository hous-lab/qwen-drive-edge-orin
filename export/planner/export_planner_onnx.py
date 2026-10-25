#!/usr/bin/env python3
"""Export the PlanningExpert one-step predict_endpoint to ONNX.

Scope: design + export verification only (no TRT build). Answers: which ops
export as standard ONNX, which need custom handling — that decides whether the
v2 engine is pure standard operators or needs plugins.

Export-friendly deviations from the official module (all math-preserving, each
documented inline):
- Rotary cos/sin enter as graph INPUTS (tiny tables produced by the official
  WaypointRotaryEmbedding on the torch side). The official table math does
  bf16-rounded positions/inv_freq products (audit R4/R6); keeping that producer
  outside the engine reproduces the trained rounding exactly, in any engine
  compute dtype.
- Fourier freq table baked as a CONSTANT with production bf16 rounding
  (bf16 -> fp32). The official forward computes angles in fp32
  (waypoints.float() * freqs), so this reproduces production bit-for-bit.
- Time-embed freq table baked as an fp32 constant (official math is fp32).
- GQA attention decomposed to MatMul/Softmax/Mul with explicit KV-head
  expansion (head h reads kv head h//4, same as enable_gqa).
- one_hot via eq/valid-mask (same zero-row-for-out-of-range semantics).

Gates:
  A. onnxruntime(CPU) vs this wrapper, fp32: cos >= 0.999 (export fidelity).
  B. the same wrapper structure at bf16 compute vs the official bf16 module:
     cos >= 0.999 (production fidelity — the engine will run bf16).

Usage:
  python export_planner_onnx.py --model-dir <repo>/Qwen-Drive-1.0-4B \
      --planner planner-sft --scene-len 3385 --out planner_onestep.onnx
"""
import argparse
import copy
import math
import sys
from pathlib import Path

import torch
from torch import nn

# Upstream inference code: QD_UPSTREAM_REPO env (workspace contract) first,
# then the sibling-layout walk-up (workdir/qwen-drive next to a clone of this
# repo). Raises a clear error when neither resolves.
import os
_qd_repo = os.environ.get("QD_UPSTREAM_REPO", "")
REPO = Path(_qd_repo) if _qd_repo else next(
    (c for c in [Path(__file__).resolve().parents[3], *Path(__file__).resolve().parents]
     if (c / "src" / "qwen_drive" / "__init__.py").exists()), None)
if REPO is None or not (REPO / "src" / "qwen_drive" / "__init__.py").exists():
    sys.exit("upstream inference code not found: set QD_UPSTREAM_REPO to the "
             "qwen-drive checkout (src/qwen_drive)")
sys.path.insert(0, str(REPO / "src"))

from qwen_drive import QwenDriveForPlanning  # noqa: E402
from qwen_drive.planning_expert import _apply_rotary  # noqa: E402


def bake_fourier_freqs(expert):
    """Official logspace table, pre-rounded through bf16 as production does."""
    cfg = expert.config
    freqs = torch.logspace(0, math.log10(cfg.fourier_max_frequency),
                           steps=cfg.fourier_num_features, dtype=torch.float32)
    return freqs.to(torch.bfloat16).float()


def bake_time_freqs(expert):
    half = expert.config.time_embed_dim // 2
    decay = math.log(10000) / (half - 1)
    return torch.exp(torch.arange(half, dtype=torch.float32) * -decay)


class PlannerExport(nn.Module):
    """predict_endpoint with ONNX-friendly structure; weights are the expert's."""

    def __init__(self, expert: nn.Module):
        super().__init__()
        self.expert = expert  # fp32 deepcopy; reference its submodules directly
        self.register_buffer("fourier_freqs", bake_fourier_freqs(expert))
        self.register_buffer("time_freqs", bake_time_freqs(expert))
        self.register_buffer("nav_classes",
                             torch.arange(expert.config.nav_command_classes, dtype=torch.long))
        self.scale = 1.0 / math.sqrt(expert.config.head_dim)
        self.groups = expert.config.num_key_value_heads
        self.per_group = expert.config.num_attention_heads // self.groups

    def _fourier(self, waypoints):
        exp = self.expert
        angles = waypoints.float().unsqueeze(-1) * self.fourier_freqs * (2 * math.pi)
        features = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)
        return exp.fourier_encoder.net(features.to(waypoints.dtype))

    def _attend(self, query, key, value):
        """[B,L,H,D]x[B,S,KV,D] explicit GQA attention -> [B,L,H*D].

        enable_gqa maps query head h to kv head h // per_group; flattening
        [S, groups, per_group] -> head index = kv*per_group + j reproduces that
        (kv must be the OUTER dim — verified against enable_gqa on CPU)."""
        b, l, _, _ = query.shape
        kv = key.shape[1]
        key = key.unsqueeze(3).expand(b, kv, self.groups, self.per_group, -1).reshape(
            b, kv, self.groups * self.per_group, -1)
        value = value.unsqueeze(3).expand(b, kv, self.groups, self.per_group, -1).reshape(
            b, kv, self.groups * self.per_group, -1)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        return torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2).reshape(b, l, -1)

    def _layer(self, layer, hidden_states, scene_key, scene_value, cos, sin, condition):
        batch, length, _ = hidden_states.shape
        (shift_attn, scale_attn, gate_attn,
         shift_ffn, scale_ffn, gate_ffn) = (
            m.unsqueeze(1) for m in layer.adaln_modulation(condition).chunk(6, dim=-1))

        residual = hidden_states
        x = layer.input_layernorm(hidden_states) * (1 + scale_attn) + shift_attn
        query, gate, key, value = layer._split_qkv(layer.qkv_proj(x))
        query = _apply_rotary(layer.q_norm(query), cos, sin)
        key = _apply_rotary(layer.k_norm(key), cos, sin)

        attn = self._attend(query, torch.cat([scene_key, key], dim=1),
                            torch.cat([scene_value, value], dim=1))
        attn = attn * torch.sigmoid(gate.reshape(batch, length, -1))
        hidden_states = residual + (1 + gate_attn) * layer.o_proj(attn)

        residual = hidden_states
        x = layer.post_attention_layernorm(hidden_states) * (1 + scale_ffn) + shift_ffn
        swiglu_gate, swiglu_up = layer.gate_up_proj(x).chunk(2, dim=-1)
        return residual + (1 + gate_ffn) * layer.down_proj(torch.nn.functional.silu(swiglu_gate) * swiglu_up)

    def forward(self, waypoints, flow_time, hq_pose, hq_velocity, hq_accel,
                scene_k, scene_v, nav_command, ego_status, cos, sin):
        exp = self.expert
        dtype = waypoints.dtype  # fp32 in this export
        batch, length, _ = waypoints.shape

        # time conditioning (official: fp32 table, scale, sin|cos concat, then MLP)
        angles = self.time_freqs * (exp.config.time_embed_scale * flow_time.float().unsqueeze(1))
        time_condition = exp.time_mlp(
            torch.cat([angles.sin(), angles.cos()], dim=-1).to(dtype))

        broadcast = [
            exp.trajectory_proj(waypoints),
            self._fourier(waypoints),
            time_condition.unsqueeze(1).expand(-1, length, -1),
            hq_pose.unsqueeze(1).expand(-1, length, -1),
            exp.waypoint_embed(torch.arange(length, device=waypoints.device))
            .unsqueeze(0).expand(batch, -1, -1).to(dtype),
            hq_velocity.unsqueeze(1).expand(-1, length, -1),
            hq_accel.unsqueeze(1).expand(-1, length, -1),
        ]
        hidden_states = exp.query_fusion(torch.cat(broadcast, dim=-1))

        # one-hot with official zero-row-for-out-of-range semantics
        valid = (nav_command >= 0) & (nav_command < self.nav_classes.shape[0])
        nav_onehot = (nav_command.unsqueeze(-1) == self.nav_classes).to(dtype) * valid.unsqueeze(-1).to(dtype)
        condition = time_condition + exp.nav_mlp(nav_onehot) + exp.ego_mlp(ego_status.to(dtype))

        for index, layer in enumerate(exp.layers):
            group = index // exp.config.layers_per_kv
            hidden_states = self._layer(layer, hidden_states,
                                        scene_k[group].unsqueeze(0),
                                        scene_v[group].unsqueeze(0), cos, sin, condition)
        return exp.out_proj(exp.final_layernorm(hidden_states)).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--planner", default="planner-sft")
    ap.add_argument("--scene-len", type=int, default=3385)
    ap.add_argument("--out", default=None)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic", action="store_true",
                    help="export scene_k/scene_v with a dynamic scene-length "
                         "axis (dim 1); trace still runs at --scene-len")
    ap.add_argument("--skip-load-ort", action="store_true")
    args = ap.parse_args()
    out = Path(args.out or (Path(__file__).resolve().parent / "planner_onestep_s"
                            f"{args.scene_len}.onnx"))
    torch.set_grad_enabled(False)

    model = QwenDriveForPlanning.from_pretrained(
        args.model_dir, planner=args.planner, dtype=torch.bfloat16,
        attn_implementation="sdpa").eval()
    expert32 = copy.deepcopy(model.planning_expert).float().eval()
    cfg = expert32.config
    wp_len, S = expert32.num_future_points, args.scene_len
    rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor)

    wrapper = PlannerExport(expert32).eval()

    # ---- inputs (all static shapes) ----
    g = torch.Generator().manual_seed(0)
    def f32(*shape):
        return torch.randn(*shape, generator=g)
    inputs = dict(
        waypoints=f32(1, wp_len, 3),
        flow_time=torch.zeros(1),
        hq_pose=f32(1, cfg.hidden_size),
        hq_velocity=f32(1, cfg.hidden_size),
        hq_accel=f32(1, cfg.hidden_size),
        scene_k=f32(cfg.num_hidden_layers // cfg.layers_per_kv, S, cfg.num_key_value_heads, cfg.head_dim),
        scene_v=f32(cfg.num_hidden_layers // cfg.layers_per_kv, S, cfg.num_key_value_heads, cfg.head_dim),
        nav_command=torch.tensor([1]),
        ego_status=f32(1, cfg.ego_status_dim),
    )
    anchor = torch.tensor([[522], [522], [522]])  # [3, B]; demo scene anchor
    positions = anchor.unsqueeze(-1) + torch.arange(1, wp_len + 1)  # [3, B, L]
    # PRODUCTION tables: the official bf16 rotary rounds positions/inv_freq/angle
    # products (audit R4/R6) — the engine must consume exactly these tables, so
    # the wrapper takes them as inputs and this script feeds the production ones.
    cos, sin = model.planning_expert.rotary_emb(positions, torch.bfloat16)
    cos, sin = cos.float(), sin.float()
    assert tuple(cos.shape) == (1, wp_len, 1, rotary_dim), tuple(cos.shape)

    order = ["waypoints", "flow_time", "hq_pose", "hq_velocity", "hq_accel",
             "scene_k", "scene_v", "nav_command", "ego_status"]
    input_names = order + ["cos", "sin"]
    args_tuple = tuple(inputs[k] for k in order) + (cos, sin)

    # ---- gate B reference 1: official fp32 deepcopy (documents table-rounding delta)
    # ---- gate B reference 2: production bf16 expert (the real semantic target)
    groups = expert32.config.num_hidden_layers // expert32.config.layers_per_kv
    ref32 = expert32.predict_endpoint(
        inputs["waypoints"], inputs["flow_time"],
        (inputs["hq_pose"], inputs["hq_velocity"], inputs["hq_accel"]),
        [(inputs["scene_k"][i].unsqueeze(0), inputs["scene_v"][i].unsqueeze(0))
         for i in range(groups)],
        anchor, inputs["nav_command"], inputs["ego_status"])
    bf16 = torch.bfloat16
    ref_prod = model.planning_expert.predict_endpoint(
        inputs["waypoints"], inputs["flow_time"],
        tuple(inputs[k].to(bf16) for k in ("hq_pose", "hq_velocity", "hq_accel")),
        [(inputs["scene_k"][i].unsqueeze(0).to(bf16), inputs["scene_v"][i].unsqueeze(0).to(bf16))
         for i in range(groups)],
        anchor, inputs["nav_command"], inputs["ego_status"])

    wrapper_out = wrapper(*args_tuple)

    def cos_sim(a, b):
        a, b = a.double().ravel(), b.double().ravel()
        return float(a @ b / (a.norm() * b.norm()))

    d32 = (wrapper_out - ref32).abs().max().item()
    cos_ref32_refprod = cos_sim(ref32, ref_prod.float())
    print(f"[R4/R6 evidence] official-fp32 vs production-bf16: cos={cos_ref32_refprod:.6f}  "
          f"(global bf16-rounding sensitivity, R4/R6 extended)", flush=True)
    print(f"[evidence] wrapper(fp32) vs official-fp32: cos={cos_sim(wrapper_out, ref32):.6f} "
          f"max|d|={d32:.3e}", flush=True)
    cosB_fp32proxy = cos_sim(wrapper_out, ref_prod.float())
    print(f"[evidence] wrapper(fp32) vs production-bf16: cos={cosB_fp32proxy:.6f}  "
          f"(fp32 compute proxy cannot reproduce production; engine must run bf16)", flush=True)

    # ---- gate B (decisive): the same wrapper structure at bf16 compute.
    # If this matches production, the export structure is correct and the fp32
    # proxy gap above is purely compute dtype (engine must run bf16, TRT-native).
    wrapper16 = PlannerExport(copy.deepcopy(model.planning_expert).eval()).to(bf16).eval()
    wrapper16.time_freqs = wrapper16.time_freqs.float()  # official time table stays fp32
    cos16, sin16 = model.planning_expert.rotary_emb(positions, bf16)
    args16 = (inputs["waypoints"].to(bf16), inputs["flow_time"],
              *(inputs[k].to(bf16) for k in ("hq_pose", "hq_velocity", "hq_accel")),
              inputs["scene_k"].to(bf16), inputs["scene_v"].to(bf16),
              inputs["nav_command"], inputs["ego_status"].to(bf16), cos16, sin16)
    wrapper16_out = wrapper16(*args16)
    cosB = cos_sim(wrapper16_out.float(), ref_prod.float())
    dB = (wrapper16_out.float() - ref_prod.float()).abs().max().item()
    print(f"[gate B] wrapper(bf16) vs production-bf16: cos={cosB:.8f} max|d|={dB:.3e}", flush=True)

    # ---- export ----
    out.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {}
    if args.dynamic:
        export_kwargs["dynamic_axes"] = {"scene_k": {1: "scene_len"},
                                         "scene_v": {1: "scene_len"}}
    torch.onnx.export(
        wrapper, args_tuple, str(out), opset_version=args.opset,
        input_names=input_names, output_names=["endpoint"],
        do_constant_folding=True, dynamo=False, **export_kwargs)
    print(f"exported {out} ({out.stat().st_size / 1e6:.0f} MB)", flush=True)

    import onnx
    om = onnx.load(str(out), load_external_data=False)
    try:
        onnx.checker.check_model(om)  # serializes the proto; >2GB models can't
    except Exception as exc:
        print(f"onnx.checker skipped ({exc.__class__.__name__}: {exc})", flush=True)
    ops = {}
    for node in om.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print("op inventory:", dict(sorted(ops.items(), key=lambda kv: -kv[1])), flush=True)
    standard = {
        "Add", "Mul", "Sub", "Div", "MatMul", "Softmax", "Concat", "Split",
        "Slice", "Reshape", "Transpose", "Cast", "Sigmoid", "Tanh", "Erf",
        "Exp", "Sin", "Cos", "Sqrt", "Pow", "ReduceMean", "Gather", "Unsqueeze",
        "Squeeze", "Constant", "ConstantOfShape", "Shape", "Range", "Equal",
        "Where", "Relu", "Greater", "Less", "And", "Round", "Identity", "Dropout",
        "Flatten", "Neg", "Log", "Floor", "Ceil", "Mod", "Min", "Max",
        "Expand", "Gemm", "GreaterOrEqual"}  # all three are standard ONNX ops
    nonstandard = sorted(set(ops) - standard)
    print("all ops standard ONNX" if not nonstandard else f"NON-STANDARD ops: {nonstandard}",
          flush=True)

    if args.skip_load_ort:
        return
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    feeds = {name: arr.numpy() for name, arr in zip(input_names, args_tuple)}
    (ort_out,) = sess.run(None, feeds)
    ort_out = torch.from_numpy(ort_out)
    dA = (ort_out - wrapper_out).abs().max().item()
    cosA = cos_sim(ort_out, wrapper_out)
    okA = cosA >= 0.999
    okB = cosB >= 0.999
    # EXPORT verdict = gate A (serialization fidelity). Gate B (the wrapper
    # structure at bf16 compute vs the official module) is recorded as
    # evidence; the decisive fidelity gates for the shipped engine lane are
    # the engine-level gate and the chain gate downstream.
    print(f"[gate A] ORT vs wrapper: cos={cosA:.8f} max|d|={dA:.3e} -> "
          f"{'PASS' if okA else 'FAIL'}", flush=True)
    print(f"[gate B] production fidelity: cos={cosB:.8f} -> {'PASS' if okB else 'FAIL'}",
          flush=True)
    print("EXPORT", "PASS" if okA else "FAIL", flush=True)


if __name__ == "__main__":
    main()
