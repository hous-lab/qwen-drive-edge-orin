#!/usr/bin/env python3
"""Dump the byte-exact TRT gate bundle for the planner one-step engine.

All floating inputs are pre-quantized to bf16 then stored; the Orin gate script
upcasts to fp32 for the engine's fp32 I/O, so ANY engine precision (bf16/fp16
tactics) consumes exactly the production values. The tables and scene KV carry
the load-bearing rounding (export gate B evidence: fp32 tables vs production
cos=0.9897, production tables + fp32 compute cos=0.9999).

References stored alongside:
  ref_prod       official production bf16 expert output (the semantic target)
  wrapper16_out  export wrapper at bf16 compute (export gate B)

Input construction MUST stay in sync with export_planner_onnx.main
(seed 0 generator, same call order, anchor 522).
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

import os
_qd_repo = os.environ.get("QD_UPSTREAM_REPO", "")
REPO = Path(_qd_repo) if _qd_repo else next(
    (c for c in [Path(__file__).resolve().parents[3], *Path(__file__).resolve().parents]
     if (c / "src" / "qwen_drive" / "__init__.py").exists()), None)
if REPO is None or not (REPO / "src" / "qwen_drive" / "__init__.py").exists():
    sys.exit("upstream inference code not found: set QD_UPSTREAM_REPO to the "
             "qwen-drive checkout (src/qwen_drive)")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_drive import QwenDriveForPlanning  # noqa: E402
from export_planner_onnx import PlannerExport  # noqa: E402

BF16 = torch.bfloat16


def cos_sim(a, b):
    a, b = a.double().ravel(), b.double().ravel()
    return float(a @ b / (a.norm() * b.norm()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--planner", required=True)
    ap.add_argument("--scene-len", type=int, default=3385)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or (Path(__file__).resolve().parent / "planner_gate_bundle.npz"))
    torch.set_grad_enabled(False)

    model = QwenDriveForPlanning.from_pretrained(
        args.model_dir, planner=args.planner, dtype=BF16,
        attn_implementation="sdpa").eval()
    expert = model.planning_expert
    cfg = expert.config
    wp_len, S = expert.num_future_points, args.scene_len

    # ---- inputs: identical to export_planner_onnx.main (seed 0, same order) ----
    g = torch.Generator().manual_seed(0)
    def f32(*shape):
        return torch.randn(*shape, generator=g)
    inputs = dict(
        waypoints=f32(1, wp_len, 3),
        flow_time=torch.zeros(1),
        hq_pose=f32(1, cfg.hidden_size),
        hq_velocity=f32(1, cfg.hidden_size),
        hq_accel=f32(1, cfg.hidden_size),
        scene_k=f32(cfg.num_hidden_layers // cfg.layers_per_kv, S,
                    cfg.num_key_value_heads, cfg.head_dim),
        scene_v=f32(cfg.num_hidden_layers // cfg.layers_per_kv, S,
                    cfg.num_key_value_heads, cfg.head_dim),
        nav_command=torch.tensor([1]),
        ego_status=f32(1, cfg.ego_status_dim),
    )
    anchor = torch.tensor([[522], [522], [522]])  # [3, B]; demo scene anchor
    positions = anchor.unsqueeze(-1) + torch.arange(1, wp_len + 1)
    cos16, sin16 = expert.rotary_emb(positions, BF16)  # production bf16 tables

    # ---- reference 1: official production bf16 expert ----
    groups = cfg.num_hidden_layers // cfg.layers_per_kv
    ref_prod = expert.predict_endpoint(
        inputs["waypoints"], inputs["flow_time"],
        tuple(inputs[k].to(BF16) for k in ("hq_pose", "hq_velocity", "hq_accel")),
        [(inputs["scene_k"][i].unsqueeze(0).to(BF16),
          inputs["scene_v"][i].unsqueeze(0).to(BF16)) for i in range(groups)],
        anchor, inputs["nav_command"], inputs["ego_status"].to(BF16)).float()

    # ---- reference 2: export wrapper at bf16 compute (gate B twin) ----
    wrapper16 = PlannerExport(copy.deepcopy(expert).eval()).to(BF16).eval()
    wrapper16.time_freqs = wrapper16.time_freqs.float()
    args16 = (inputs["waypoints"].to(BF16), inputs["flow_time"],
              *(inputs[k].to(BF16) for k in ("hq_pose", "hq_velocity", "hq_accel")),
              inputs["scene_k"].to(BF16), inputs["scene_v"].to(BF16),
              inputs["nav_command"], inputs["ego_status"].to(BF16), cos16, sin16)
    wrapper16_out = wrapper16(*args16).float()

    c12 = cos_sim(ref_prod, wrapper16_out)
    print(f"[bundle] ref_prod vs wrapper16_out: cos={c12:.8f} (expect ~0.99997, gate B)", flush=True)

    def q(t):  # pre-quantize to production bf16; numpy has no bf16, so store
        return t.to(BF16).float()  # the exact fp32 upcast of the quantized values

    arrays = {
        "waypoints": q(inputs["waypoints"]).numpy(),
        "flow_time": inputs["flow_time"].numpy(),
        "hq_pose": q(inputs["hq_pose"]).numpy(),
        "hq_velocity": q(inputs["hq_velocity"]).numpy(),
        "hq_accel": q(inputs["hq_accel"]).numpy(),
        "scene_k": q(inputs["scene_k"]).numpy(),
        "scene_v": q(inputs["scene_v"]).numpy(),
        "nav_command": inputs["nav_command"].numpy(),
        "ego_status": q(inputs["ego_status"]).numpy(),
        "cos": cos16.float().numpy(), "sin": sin16.float().numpy(),
        "ref_prod": ref_prod.numpy(), "wrapper16_out": wrapper16_out.numpy(),
    }
    np.savez(out, **arrays)
    mb = out.stat().st_size / 1e6
    print(f"[bundle] wrote {out} ({mb:.0f} MB); scene S={S}, wp={wp_len}", flush=True)


if __name__ == "__main__":
    main()
