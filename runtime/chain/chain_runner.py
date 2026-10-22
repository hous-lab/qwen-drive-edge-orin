#!/usr/bin/env python3
"""Full-engine-chain runner: processor -> TRT vision -> TRT text -> planner.

Extends golden_runner's protocol with two engine-in-loop substitutions:

- ``--vision-engine``: the existing install_vision_engine patch (visual tower
  -> TRT plan, grid baked at trace time, loud failure on grid deviation).
- ``--text-engine``: patches ``model.vlm.forward`` to run the Edge-LLM text
  engine on merged inputs_embeds. The engine consumes a static rope table
  ``[1, kvLen, 64]``; production M-RoPE positions (get_rope_index, [3,B,S])
  are folded into per-token composite rows (mrope_rows.py, self-tested
  against the torch rotary module). KV past buffers double as present
  (attentionPlugin WAR alias), so scene_cache is sliced straight out of them
  and handed to the expert in exactly the _scene_cache layout.
- ``--graph-planner``: the existing install_cuda_graph_planner (v1.1
  bucketing). Note its signature check requires scene_cache dtype ==
  expert.dtype, so the engine's fp16 KV views are cast once here.

Staged integration: run with no engine flags = golden parity harness (useful
as the same-device baseline writer); enable one engine at a time and compare
with eval_chain.py (traj mean|d| <= 5cm + cos >= 0.999 vs the golden arm).
"""
import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _find_repo():
    """Locate the upstream qwen-drive repo (provides the qwen_drive package).

    QD_UPSTREAM_REPO env var wins; otherwise walk up from this file.
    """
    env = os.environ.get("QD_UPSTREAM_REPO")
    cands = [Path(env)] if env else []
    for cand in [*cands, *Path(__file__).resolve().parents]:
        if (cand / "src" / "qwen_drive" / "__init__.py").exists():
            return cand
    raise RuntimeError(
        "qwen_drive package not found: set QD_UPSTREAM_REPO to the checkout of "
        "the upstream qwen-drive repo (see ckpts/README.md)")

REPO = _find_repo()
sys.path.insert(0, str(REPO / "src"))
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


from qwen_drive import QwenDriveForPlanning  # noqa: E402
from qwen_drive.benchmarks import read_scene_file  # noqa: E402
from qwen_drive.images import ImageArchive  # noqa: E402


def _torch_dtype(trt_dtype):
    """trt.DataType -> torch dtype (numpy has no bf16, so trt.nptype is unsafe)."""
    import tensorrt as trt
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
        trt.DataType.INT8: torch.int8,
        trt.DataType.UINT8: torch.uint8,
    }
    try:
        return mapping[trt_dtype]
    except KeyError:
        raise ValueError(f"unsupported TRT dtype {trt_dtype}")
from mrope_rows import arange_rows, mrope_rows  # noqa: E402

def _plugin_so():
    so = os.environ.get("QD_PLUGIN_SO")
    if not so:
        raise RuntimeError(
            "QD_PLUGIN_SO is not set: point it at the Edge-LLM plugin library "
            "libNvInfer_edgellm_plugin.so built from "
            "export/patches/orin_text_engine/ (see kernels/README.md)")
    return so

_FULL_ATTN = [3, 7, 11, 15, 19, 23, 27, 31]  # layer_types, interval 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--planner", required=True)
    p.add_argument("--scenes", required=True)
    p.add_argument("--image-root", default=None)
    p.add_argument("--image-archive", default=None)
    p.add_argument("--mode", choices=["direct", "reasoning"], default="direct")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--vision-engine", default=None,
                   help="TRT .plan replacing the vlm visual tower")
    p.add_argument("--text-engine", default=None,
                   help="TRT .engine replacing the vlm text prefill")
    p.add_argument("--kvlen", type=int, default=4096,
                   help="KV capacity fed to past_key_values_* (engine dim)")
    p.add_argument("--zero-kv", dest="zero_kv", action="store_true", default=True,
                   help="re-zero KV capacity + GDN states before every scene")
    p.add_argument("--no-zero-kv", dest="zero_kv", action="store_false")
    p.add_argument("--rope-mode", choices=["mrope", "arange", "zero"],
                   default="mrope",
                   help="debug: rope table content (mrope composite / arange / "
                        "zero=identity)")
    p.add_argument("--graph-planner", action="store_true")
    p.add_argument("--planner-engine", dest="planner_engine", action="append",
                   default=None,
                   help="TRT .plan replacing PlanningExpert.predict_endpoint "
                        "(repeat flag for one engine per static scene-KV length)")
    p.add_argument("--graph-buffers", dest="graph_buffers", action="store_true",
                   default=True)
    p.add_argument("--no-graph-buffers", dest="graph_buffers", action="store_false")
    p.add_argument("--dump-cache-dir", default=None,
                   help="npz per scene: embeds/positions/rope rows (engine arm) "
                        "or scene_cache+anchor (both arms), first N scenes")
    p.add_argument("--dump-scenes", type=int, default=2)
    return p.parse_args()


def stage_time(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, (time.perf_counter() - t0) * 1e3


class _Layer:
    """Minimal past_key_values layer: .keys/.values [1, heads, seq, dim]."""
    def __init__(self, keys=None, values=None):
        self.keys, self.values = keys, values


class _FakeCache:
    def __init__(self, layers):
        self.layers = layers  # list indexed by ABSOLUTE layer index


class TextEngineRunner:
    """Raw TRT driver for the Edge-LLM prefill engine on torch buffers.

    Contract (gate/README.md, all gate-verified): inputs_embeds [1,S,2560]
    fp16; past_key_values_{0..7} [1,2,Hkv,KVLEN,hd] fp16 with dim 4 =
    CAPACITY; present_* aliases the past buffer (WAR in-place); rope
    [1,KVLEN,64] fp32 STATIC, row p = cos||sin for whatever we put at p
    (arange in the gates, M-RoPE composite here); conv_state_{0..23} /
    recurrent_state_{0..23} zeroed; context_lengths/kvcache_start_index i32,
    last_token_ids i64. Outputs: logits [1,1,V] fp32 + present_*.
    """

    def __init__(self, engine_path, kvlen=4096, zero_kv=True, rope_mode="mrope"):
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        lib = ctypes.CDLL(_plugin_so(), mode=ctypes.RTLD_GLOBAL)
        lib.initEdgellmPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.initEdgellmPlugins.restype = ctypes.c_bool
        assert lib.initEdgellmPlugins(None, b""), "initEdgellmPlugins failed"
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        assert self.engine is not None, f"deserialize failed: {engine_path}"
        self.ctx = self.engine.create_execution_context()

        self.kvlen = kvlen
        self.zero_kv = zero_kv
        self.rope_mode = rope_mode
        self.device = torch.device("cuda:0")
        self.timings = {}

        self.inputs, self.outputs = {}, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            base = name.split("[")[0].strip()
            mode = self.engine.get_tensor_mode(name)
            dt = _torch_dtype(self.engine.get_tensor_dtype(name))
            (self.inputs if mode == trt.TensorIOMode.INPUT else self.outputs)[base] = \
                (name, dt)

        # static rope table: arange rows; M-RoPE rows overwrite [:T] per call
        rot = 64
        for name, _ in self.inputs.values():
            if "rope" in name:
                shape = tuple(int(x) for x in self.ctx.get_tensor_shape(name))
                rot = shape[-1]
        self.rot = rot
        # C-contiguous is LOAD-BEARING (bisect 2026-10-06): the engine reads
        # this buffer raw row-major. arange_rows came back F-contiguous
        # (transpose view), from_numpy/.to() preserve strides, and the engine
        # consumed a transposed scramble of the table -> KV cos 0.66, while
        # every torch-side readback of the logical content looked correct.
        table = torch.from_numpy(np.ascontiguousarray(
            arange_rows(kvlen, rotary_dim=rot)))
        self.rope = table.to(self.device).contiguous()        # [kvlen, 64] fp32
        assert self.rope.is_contiguous(), "rope table must be C-contiguous"
        if rope_mode == "zero":
            self.rope.zero_()                                # identity rotation
        self._rope_stale_upto = 0

        # KV capacity buffers + states, all zeros.
        # Binding discipline: ALL input shapes first, THEN all addresses
        # (matches the text-engine2 probe flow that gates at 0.9998).
        # NOTE (bisect 2026-10-06): the historical "interleaved order scrambles
        # KV" attribution was wrong — the real defect was the F-contiguous
        # rope table consumed transposed by the engine (see table init above).
        n_layers = sum(1 for b in self.inputs if b.startswith("past_key_values_"))
        self.max_len = kvlen
        assert n_layers == 8, f"expected 8 KV groups, engine has {n_layers}"
        hidden = self._static_dim("inputs_embeds", 2)

        def _input_shape(base, shape):
            name, _ = self.inputs[base]
            ok = self.ctx.set_input_shape(name, shape)
            assert ok, f"set_input_shape {base} {shape}"

        # pass A: every input shape
        _input_shape("inputs_embeds", (1, kvlen, hidden))
        if "rope_rotary_cos_sin" in self.inputs:
            _input_shape("rope_rotary_cos_sin", (1, kvlen, self.rot))
        for base, (name, dt) in self.inputs.items():
            shape = tuple(int(x) for x in self.ctx.get_tensor_shape(name))
            if base.startswith("past_key_values_"):
                shape = tuple(kvlen if s == -1 else s for s in shape)
            elif base.startswith(("conv_state_", "recurrent_state_")):
                pass
            elif base == "context_lengths" or base == "kvcache_start_index":
                shape = (1,)
            elif base == "last_token_ids":
                shape = (1, 1)
            elif base == "inputs_embeds" or "rope" in base:
                continue                      # set explicitly above
            else:
                continue                      # unknown extra input: leave static
            ok = self.ctx.set_input_shape(name, shape)
            assert ok, f"set_input_shape {base} {shape}"

        # pass B: every input buffer + address (aliases included)
        self.ctx_lens = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.ctx_start = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.last_ids = torch.zeros(1, 1, dtype=torch.int64, device=self.device)
        self.kv = {}
        self.states = []
        for base, (name, dt) in self.inputs.items():
            if base.startswith("past_key_values_"):
                shape = tuple(kvlen if s == -1 else s
                              for s in self.ctx.get_tensor_shape(name))
                buf = torch.zeros(shape, dtype=dt, device=self.device)
                self.kv[base] = buf
            elif base.startswith(("conv_state_", "recurrent_state_")):
                shape = tuple(int(x) for x in self.ctx.get_tensor_shape(name))
                buf = torch.zeros(shape, dtype=dt, device=self.device)
                self.states.append((name, buf))
            elif base == "inputs_embeds":
                self.embeds = torch.zeros(1, self.max_len, hidden,
                                          dtype=torch.float16, device=self.device)
                buf = self.embeds
            elif base == "context_lengths":
                buf = self.ctx_lens
            elif base == "kvcache_start_index":
                buf = self.ctx_start
            elif base == "last_token_ids":
                buf = self.last_ids
            elif "rope" in base:
                buf = self.rope
            else:
                continue
            self.ctx.set_tensor_address(name, buf.data_ptr())
            if base.startswith("past_key_values_"):
                present = self.outputs.get(base.replace("past_", "present_"))
                if present:
                    self.ctx.set_tensor_address(present[0], buf.data_ptr())

        # pass C: outputs not aliased to a past_* KV buffer get their own
        # buffer; present_conv_state_*/present_recurrent_state_* are real
        # outputs and MUST be bound (enqueueV3 fails otherwise)
        kv_out_names = {b.replace("past_", "present_") for b in self.kv}
        self.logits = None
        for base, (name, dt) in list(self.outputs.items()):
            if base in kv_out_names:
                continue
            shape = tuple(kvlen if s == -1 else s
                          for s in self.ctx.get_tensor_shape(name))
            buf = torch.zeros(shape, dtype=dt, device=self.device)
            self.ctx.set_tensor_address(name, buf.data_ptr())
            if base == "logits":
                self.logits = buf
        assert self.logits is not None, "engine has no logits output"

    def _static_dim(self, base, axis):
        name, _ = self.inputs[base] if base in self.inputs else self.outputs[base]
        shape = tuple(int(x) for x in self.ctx.get_tensor_shape(name))
        return shape[axis]

    def prefill(self, embeds_bf16, position_ids):
        """embeds_bf16 [T,hidden] on device; position_ids [3,T] (T,H,W).

        Returns (fake cache over the 8 KV groups [1,Hkv,T,hd] each in engine
        fp16, logits [1,1,V]).
        """
        t0 = time.perf_counter()
        t_len = embeds_bf16.shape[0]
        assert t_len <= self.max_len, f"T={t_len} > kvlen={self.max_len}"
        name, _ = self.inputs["inputs_embeds"]
        ok = self.ctx.set_input_shape(name, (1, t_len, embeds_bf16.shape[1]))
        assert ok, "set_input_shape inputs_embeds"
        self.embeds[0, :t_len].copy_(embeds_bf16.to(torch.float16))
        self.ctx_lens.fill_(t_len)
        self.last_ids.fill_(t_len - 1)

        # M-RoPE composite rows for tokens [0, T); arange rows beyond stay
        if self.rope_mode == "mrope":
            rows = torch.from_numpy(
                mrope_rows(position_ids.detach().cpu().numpy()))
            assert rows.is_contiguous(), "mrope rows must be C-contiguous"
            self.rope[:t_len].copy_(rows.to(self.device))

        if self.zero_kv:
            for _, buf in self.states:
                buf.zero_()
            if self.zero_kv:
                for buf in self.kv.values():
                    buf.zero_()
        torch.cuda.synchronize()
        self.timings["stage_in_ms"] = (time.perf_counter() - t0) * 1e3

        t1 = time.perf_counter()
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TRT execute_async_v3 returned False")
        torch.cuda.synchronize()
        self.timings["engine_ms"] = (time.perf_counter() - t1) * 1e3

        layers = [None] * 32
        for g in range(8):
            buf = self.kv[f"past_key_values_{g}"]        # [1,2,Hkv,cap,hd]
            key = buf[0, 0, :, :t_len, :].unsqueeze(0)   # [1,Hkv,T,hd]
            val = buf[0, 1, :, :t_len, :].unsqueeze(0)
            layers[_FULL_ATTN[g]] = _Layer(key, val)
        return _FakeCache(layers), self.logits


def install_text_engine(model, engine_path, kvlen=4096, zero_kv=True,
                        rope_mode="mrope"):
    """Patch model.vlm.forward: torch-merge embeds, engine prefill, fake cache."""
    runner = TextEngineRunner(engine_path, kvlen=kvlen, zero_kv=zero_kv,
                              rope_mode=rope_mode)
    vlm = model.vlm
    inner = vlm.model
    image_token_id = model.config.vlm_config.image_token_id
    expert_dtype = model.planning_expert.dtype
    state = {"dump": None}

    def forward(input_ids=None, pixel_values=None, image_grid_thw=None,
                mm_token_type_ids=None, use_cache=True, **_):
        t0 = time.perf_counter()
        emb = inner.get_input_embeddings()(input_ids)          # [1,S,2560]
        feats = None
        if pixel_values is not None:
            out = inner.get_image_features(
                pixel_values, image_grid_thw, return_dict=True)
            feats = out.pooler_output
            if isinstance(feats, (list, tuple)):
                feats = torch.cat(feats, dim=0)
            feats = feats.to(dtype=emb.dtype, device=emb.device)
        torch.cuda.synchronize()
        runner.timings["vision_ms"] = (time.perf_counter() - t0) * 1e3

        t1 = time.perf_counter()
        if feats is not None:
            mask = input_ids == image_token_id
            n_img = int(mask.sum())
            assert n_img == feats.shape[0], \
                f"image tokens {n_img} != features {feats.shape[0]}"
            # HF pattern: mask must cover the embedding dim for masked_scatter
            emb = emb.masked_scatter(
                mask[:, :, None].expand_as(emb), feats)
        positions, _ = inner.get_rope_index(
            input_ids, mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw)                     # [3,1,S]
        if state["dump"] is not None:
            state["dump"](emb, positions)
        cache, logits = runner.prefill(emb[0], positions[:, 0])
        # planner signature check wants expert.dtype scene_cache
        layers = [_Layer(l.keys.to(expert_dtype), l.values.to(expert_dtype))
                  if l is not None else None for l in cache.layers]
        torch.cuda.synchronize()
        runner.timings["text_ms"] = (time.perf_counter() - t1) * 1e3
        return SimpleNamespace(past_key_values=_FakeCache(layers), logits=logits)

    vlm.forward = forward
    runner._dump_state = state
    return runner


def install_cache_dump(model, out_dir, max_scenes):
    """Wrap _scene_cache to save per-scene KV + anchor (works for both arms)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    orig = model._scene_cache
    state = {"n": 0}

    def wrapped(past_key_values):
        scene_cache = orig(past_key_values)          # list of (k, v) per group
        if state["n"] < max_scenes:
            arrays = {}
            for g, (k, v) in enumerate(scene_cache):
                arrays[f"k_{g}"] = k[0].float().cpu().numpy()
                arrays[f"v_{g}"] = v[0].float().cpu().numpy()
            np.savez(out / f"cache_{state['n']:04d}.npz", **arrays)
            state["n"] += 1
        return scene_cache

    model._scene_cache = wrapped


def main():
    args = parse_args()
    if args.text_engine and args.graph_planner:
        # KNOWN DEFECT (2026-10-06): CUDA-graph planner corruption under TRT
        # engine co-residency. With >=2 prior graph captures in an engine
        # process, a later graph's replays #2..10 produce deterministic garbage
        # (r640 scene 0391b4a7: 242.67cm vs eager 0.70cm; capture-time replay
        # is bitwise-equal, pool sharing / graph resets / fresh contexts do not
        # help). All-torch multi-capture runs are bitwise-clean. Refuse the
        # combination rather than emit silently corrupted trajectories.
        sys.exit("chain_runner: --text-engine + --graph-planner is unsupported "
                 "(planner graph corruption under engine co-residency, see "
                 "06_planner_accel/planner_cuda_graph.py header); "
                 "drop --graph-planner (eager planner is gate-clean)")
    if args.vision_engine and args.text_engine:
        # KNOWN DEFECT (2026-10-06): out-of-bounds GPU write when BOTH TRT
        # engines are co-resident. Small torch tensors get overwritten with
        # random 64-bit garbage (observed victim: inputs["image_grid_thw"],
        # 288 B int64) and crash a few scenes in (arange(llm_grid_t<0) in
        # get_rope_index). Victim index + tensor wander with heap layout
        # (heisenbug: clone instrumentation makes it vanish); each engine
        # alone ran 94 scenes clean; private streams + event fences do not
        # fix it; compute-sanitizer is unsupported on this Orin iGPU so the
        # writing kernel cannot be identified here. Repro: scenes 1-7 of
        # chunk0_r640 with both engines. Refuse rather than emit silently
        # corrupted trajectories; unblock after an upstream TRT/Edge-LLM fix.
        sys.exit("chain_runner: --vision-engine + --text-engine in ONE process "
                 "is unsupported (co-residency OOB write corrupts small torch "
                 "tensors; run the engines in separate processes / separate "
                 "arms until fixed upstream)")
    if args.planner_engine and args.graph_planner:
        # Both patch expert.predict_endpoint — the engine would be shadowed by
        # whichever install runs last (and blocker 1 forbids graphs under
        # engines anyway). Refuse rather than run a silently wrong arm.
        sys.exit("chain_runner: --planner-engine + --graph-planner are mutually "
                 "exclusive (both patch predict_endpoint)")
    if args.planner_engine and args.text_engine:
        # A THIRD TRT engine alongside the text engine — same multi-engine
        # co-residency family as blocker 2 (vision+text OOB write). The
        # sanctioned experiment is vision-engine + planner-engine.
        sys.exit("chain_runner: --planner-engine + --text-engine is unsupported "
                 "(three-engine co-residency, blocker-2 family; vision+planner "
                 "is the sanctioned combination)")
    torch.set_grad_enabled(False)
    device = "cuda:0"

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    model = QwenDriveForPlanning.from_pretrained(
        args.model_dir, planner=args.planner, dtype=dtype,
        attn_implementation="sdpa").to(device).eval()

    text_runner = None
    planner_state = None
    if args.vision_engine:
        from vision_engine_inloop import install_vision_engine
        install_vision_engine(model, args.vision_engine)
        print(f"vision engine in-loop: {Path(args.vision_engine).name}", flush=True)
    if args.planner_engine:
        from planner_engine_inloop import install_planner_engine
        planner_state = install_planner_engine(model, args.planner_engine)
        print(f"planner engine in-loop: "
              f"{[Path(p).name for p in args.planner_engine]}", flush=True)
    if args.text_engine:
        text_runner = install_text_engine(model, args.text_engine,
                                          kvlen=args.kvlen, zero_kv=args.zero_kv,
                                          rope_mode=args.rope_mode)
        print(f"text engine in-loop: {Path(args.text_engine).name} "
              f"(kvlen={args.kvlen}, zero_kv={args.zero_kv})", flush=True)
    if args.graph_planner:
        from planner_cuda_graph import install_cuda_graph_planner
        install_cuda_graph_planner(model, use_buffers=args.graph_buffers)
        print(f"cuda-graph planner installed (buffers={args.graph_buffers})",
              flush=True)
    if args.dump_cache_dir:
        install_cache_dump(model, args.dump_cache_dir, args.dump_scenes)
        if text_runner is not None:
            def _dump(emb, positions):
                d = Path(args.dump_cache_dir)
                d.mkdir(parents=True, exist_ok=True)
                np.savez(d / "embeds_0000.npz",
                         embeds=emb[0].float().cpu().numpy(),
                         positions=positions[:, 0].cpu().numpy())
            text_runner._dump_state["dump"] = _dump

    archive = ImageArchive.open(args.image_archive) if args.image_archive else None
    scenes = list(read_scene_file(args.scenes, image_root=args.image_root,
                                  image_archive=archive, limit=args.limit))

    chain = ("torch" if args.vision_engine is None else "vision-eng") + "+" + \
            ("torch" if args.text_engine is None else "text-eng") + "+" + \
            ("planner-eng" if args.planner_engine else
             "graph" if args.graph_planner else "eager")
    print(f"scenes={len(scenes)} chain={chain} steps={args.num_steps}", flush=True)

    out_f = open(args.output, "w")
    manifest = {
        "kind": "chain_run", "model": str(Path(args.model_dir).name),
        "planner": Path(args.planner).name, "mode": args.mode,
        "num_steps": args.num_steps, "num_samples": args.num_samples,
        "seed": args.seed, "chain": chain,
        "vision_engine": args.vision_engine, "text_engine": args.text_engine,
        "planner_engine": args.planner_engine,
        "graph_buffers": args.graph_buffers, "kvlen": args.kvlen,
        "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name(0),
    }
    out_f.write(json.dumps({"_manifest": manifest}) + "\n")

    totals = {"n": 0, "process_ms": 0.0, "prefill_ms": 0.0, "plan_ms": 0.0,
              "e2e_ms": 0.0, "vision_ms": 0.0, "text_ms": 0.0, "engine_ms": 0.0}
    for sample in scenes:
        t0 = time.perf_counter()
        inputs = model.processor(
            sample.scene, with_reasoning=(args.mode == "reasoning"), device="cpu")
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in inputs.items()}

        if args.mode == "reasoning":
            raise SystemExit("chain runner supports direct mode only")
        (scene_cache, anchor), ms_prefill = stage_time(lambda: model._prefill(inputs))
        (traj,), ms_plan = stage_time(lambda: model._plan_from_cache(
            scene_cache, anchor, inputs, num_samples=args.num_samples,
            num_steps=args.num_steps,
            seed=args.seed if args.seed is not None else model.config.noise_seed))

        if text_runner is not None:
            tm = text_runner.timings
            totals["vision_ms"] += tm.get("vision_ms", 0.0)
            totals["text_ms"] += tm.get("text_ms", 0.0)
            totals["engine_ms"] += tm.get("engine_ms", 0.0)

        row = {"token": sample.token, "scene_token": sample.scene_token,
               "trajectories": np.asarray(traj, dtype=np.float32).tolist()}
        if sample.future_trajectory is not None:
            row["future_trajectory"] = sample.future_trajectory.astype(float).tolist()
        ms_e2e = (time.perf_counter() - t0) * 1e3
        ms = {"prefill": round(ms_prefill, 2), "plan": round(ms_plan, 2),
              "e2e": round(ms_e2e, 2)}
        if text_runner is not None:
            for k in ("vision_ms", "text_ms", "engine_ms", "stage_in_ms"):
                v = text_runner.timings.get(k)
                if v is not None:
                    ms[k.removesuffix("_ms")] = round(v, 2)
        row["ms"] = ms
        out_f.write(json.dumps(row) + "\n")

        totals["n"] += 1
        totals["process_ms"] += ms_e2e - ms_prefill - ms_plan
        totals["prefill_ms"] += ms_prefill
        totals["plan_ms"] += ms_plan
        totals["e2e_ms"] += ms_e2e
        if totals["n"] % 10 == 0:
            print(f"  [{totals['n']}] e2e={ms_e2e:.0f}ms prefill={ms_prefill:.0f}ms "
                  f"plan={ms_plan:.0f}ms", flush=True)

    out_f.close()
    n = max(totals["n"], 1)
    extra = (f" vision={totals['vision_ms']/n:.1f} text_merge+rope={totals['text_ms']/n:.1f}"
             f" trt_text={totals['engine_ms']/n:.1f}") if text_runner else ""
    print(f"AVG_MS process={totals['process_ms']/n:.1f} "
          f"prefill={totals['prefill_ms']/n:.1f} plan={totals['plan_ms']/n:.1f} "
          f"e2e={totals['e2e_ms']/n:.1f}{extra} over {n} scenes", flush=True)
    if planner_state is not None:
        eng, egr = planner_state["engine_calls"], planner_state["eager_calls"]
        stat = (f"PLANNER_ENG engine={eng} eager_fallback={egr} "
                f"disabled={planner_state['disabled']} "
                f"eager_shapes={planner_state['eager_shapes']}")
        print(stat, flush=True)
        print(f"-> {stat}", file=sys.stderr, flush=True)
    print(f"-> {args.output}", flush=True)


if __name__ == "__main__":
    main()
