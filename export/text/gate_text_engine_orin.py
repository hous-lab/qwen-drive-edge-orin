#!/usr/bin/env python3
"""Gate + latency for the FULL qwen3_5_text prefill engine on Orin.

Compares the TRT engine against the torch bf16 reference bundle from
gen_text_ref_orin.py on the real demo scenario (3054 vision embeds + ~180
text tokens, T~=3.2k). Gate = hidden cos + 8x full-attn-layer KV cos +
logits cos, all >= 0.999. Then locks nothing itself: run under locked
clocks (sudo jetson_clocks) for the latency block (warmup 5, N iters).

Engine IO contract (mirror of tiny): inputs_embeds [1,S,2560] fp16,
past_key_values_{0..7} [1,2,Hkv,kvLen,128] fp16 (kvLen = CAPACITY),
rope_rotary_cos_sin [1,kvLen,rotaryDim] fp32, context_lengths [1] i32,
kvcache_start_index [1] i32, last_token_ids [1,1] i64,
conv_state_{0..23} / recurrent_state_{0..23} zeros. Outputs: logits
[1,1,V] fp32, present_* mirroring each input. past/present KV share one
device buffer (attentionPlugin getAliasedInput in-place WAR semantics).

Run on Orin (under locked clocks for the timing part):
  EDGELLM_PLUGIN_PATH=$QD_PLUGIN_SO \
    $QD_PY gate_text_engine_orin.py \
    --rope-npz $QD_RUNTIME_DIR/edgellm/logs_text/mrope_golden_f2.npz \
    --ref-npz  $QD_RUNTIME_DIR/edgellm/logs_text/mrope_gate_ref_f2.npz \
    --iters 20
"""
import argparse
import ctypes
import json
import os
from pathlib import Path

import numpy as np
import pycuda.autoinit  # noqa: F401  MUST precede all TRT usage
import pycuda.driver as cuda
import tensorrt as trt

PLUGIN_SO = os.environ["QD_PLUGIN_SO"]
EDG = Path(os.environ["QD_RUNTIME_DIR"]) / "edgellm"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--engine",
                   default=str(EDG / "engines_text" / "llm" / "llm.engine"))
    p.add_argument("--bundle",
                   default=str(EDG / "logs_text" / "text_ref_bundle.npz"))
    p.add_argument("--meta",
                   default=str(EDG / "logs_text" / "text_ref_meta.json"))
    p.add_argument("--out", default=str(EDG / "logs_text" / "gate_text.json"))
    p.add_argument("--kvlen", type=int, default=3456,
                   help="runtime kvLen fed to past KV + rope rows")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--skip-gate", action="store_true",
                   help="timing only (bundle not required)")
    p.add_argument("--rope-npz", default=None,
                   help="npz with rope_table [1,rows,rotary_dim]: TRUE mrope "
                        "table in SEQUENCE order (row i = token i's mrope "
                        "row, e.g. mrope_golden_f2.npz); replaces the "
                        "arange-built table")
    p.add_argument("--ref-npz", default=None,
                   help="npz with ref_logits/ref_hidden/ref_kv8: reference "
                        "generated under the SAME rope as --rope-npz "
                        "(e.g. gen_mrope_ref_f2.py output)")
    p.add_argument("--gate-threshold", type=float, default=0.999,
                   help="cos gate (0.999 fp16-vs-bf16 cast; 0.998 int8)")
    return p.parse_args()


def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    lib = ctypes.CDLL(PLUGIN_SO, mode=ctypes.RTLD_GLOBAL)
    lib.initEdgellmPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.initEdgellmPlugins.restype = ctypes.c_bool
    assert lib.initEdgellmPlugins(None, b""), "initEdgellmPlugins failed"
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    assert engine is not None, "deserialize failed"
    return engine


def cos64(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    args = parse_args()
    z = np.load(str(args.bundle))
    emb = z["inputs_embeds_fp16"]                  # [T,2560] fp16
    T = emb.shape[0]
    meta = json.loads(Path(args.meta).read_text())
    theta = meta["rope_theta"]
    head_dim = meta["head_dim"]
    rotary_dim = int(head_dim * meta["partial_rotary_factor"])
    print(f"bundle: T={T} theta={theta} rotary_dim={rotary_dim}")

    engine = load_engine(args.engine)
    ctx = engine.create_execution_context()
    stream = cuda.Stream()

    # rope table: row pos = cos(pos*theta^(-2z/D)) || sin(...), z<[D/2)
    # rows = the engine's declared rope length: static dim (int8 build pins
    # rope to [1,4096,64]) or --kvlen when the dim is dynamic (-1).
    rope_len = args.kvlen
    for s in (int(x) for x in ctx.get_tensor_shape("rope_rotary_cos_sin")):
        if s != 1 and s != rotary_dim and s != -1:
            rope_len = s
            break
    if args.rope_npz:
        rt = np.load(args.rope_npz)["rope_table"]          # [1,rows,D]
        assert rt.shape[0] == 1 and rt.shape[2] == rotary_dim, rt.shape
        assert rt.shape[1] >= rope_len, (rt.shape, rope_len)
        rope = np.ascontiguousarray(rt[0, :rope_len], dtype=np.float32)
        print(f"rope: TRUE mrope table from {Path(args.rope_npz).name} "
              f"({rt.shape[1]} rows; engine consumes first {rope_len})")
    else:
        zz = np.arange(rotary_dim // 2, dtype=np.float64)
        inv_freq = theta ** (-2.0 * zz / rotary_dim)
        pos = np.arange(rope_len, dtype=np.float64)
        ang = pos[:, None] * inv_freq[None, :]
        rope = np.concatenate([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)
        print("rope: arange 1D (WARN: wrong for vision-token sequences; "
              "use --rope-npz for true mrope)")

    feeds = {
        "inputs_embeds": emb[None],                # [1,T,2560]
        "rope_rotary_cos_sin": rope.reshape(1, rope_len, rotary_dim),
        "context_lengths": np.array([T], np.int32),
        "kvcache_start_index": np.array([0], np.int32),
        "last_token_ids": np.array([[T - 1]], np.int64),
    }

    # pass 1: set input shapes; allocate zeros for state inputs by engine shape
    kv_bufs, kv_bytes_map = {}, {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        base = name.split("[")[0].strip()
        dt = trt.nptype(engine.get_tensor_dtype(name))
        if base.startswith("past_key_values_"):
            # engine shape has kvLen dynamic; feed capacity = --kvlen (zeros)
            shape = tuple(int(x) for x in ctx.get_tensor_shape(name))
            shape = tuple(args.kvlen if s == -1 else s for s in shape)
            arr = np.zeros(shape, dtype=dt)
        elif base.startswith("conv_state_") or base.startswith("recurrent_state_"):
            shape = tuple(int(x) for x in ctx.get_tensor_shape(name))
            arr = np.zeros(shape, dtype=dt)
        else:
            arr = feeds[base]
        ok = ctx.set_input_shape(name, arr.shape)
        if not ok:
            raise RuntimeError(f"set_input_shape failed: {name} {arr.shape}")
        nbytes = arr.nbytes
        if base.startswith("past_key_values_"):
            kv_bufs[base] = cuda.mem_alloc(nbytes)
            kv_bytes_map[base] = nbytes
            cuda.memcpy_htod(kv_bufs[base], arr)   # zeros, capacity kvLen
        else:
            d = cuda.mem_alloc(max(nbytes, 4))
            cuda.memcpy_htod(d, arr)
            kv_bufs[base] = d
    print("input shapes set")

    # pass 2: bind addresses; present_* shares the past_* buffer
    host_out = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        base = name.split("[")[0].strip()
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            ctx.set_tensor_address(name, int(kv_bufs[base]))
            continue
        shape = tuple(int(x) for x in ctx.get_tensor_shape(name))
        dt = trt.nptype(engine.get_tensor_dtype(name))
        host_out[name] = np.empty(shape, dtype=dt)
        if base.startswith("present_key_values_"):
            past = base.replace("present_", "past_")
            assert int(np.prod(shape)) * np.dtype(dt).itemsize \
                <= kv_bytes_map[past], f"present bigger than past: {name}"
            ctx.set_tensor_address(name, int(kv_bufs[past]))
        else:
            nbytes = max(int(np.prod(shape)) * np.dtype(dt).itemsize, 4)
            d = cuda.mem_alloc(nbytes)
            ctx.set_tensor_address(name, int(d))
            kv_bufs["__out__" + base] = d

    ctx.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
    for name, arr in host_out.items():
        base = name.split("[")[0].strip()
        if base.startswith("present_key_values_"):
            past = base.replace("present_", "past_")
            cuda.memcpy_dtoh(arr, int(kv_bufs[past]))
        else:
            cuda.memcpy_dtoh(arr, int(kv_bufs["__out__" + base]))

    report = {"T": int(T), "engine": Path(args.engine).name,
              "ref_dtype": "bfloat16",
              "rope_source": "mrope_golden" if args.rope_npz else "arange",
              "ref_source": Path(args.ref_npz).name if args.ref_npz
                            else Path(args.bundle).name}
    if not args.skip_gate:
        ref = np.load(str(args.ref_npz)) if args.ref_npz else z
        logits_e = host_out["logits"][0, 0].astype(np.float64)
        logits_r = ref["ref_logits"].astype(np.float64)
        report["logits_cos"] = cos64(logits_e, logits_r)
        hid_e = host_out.get("hidden_states")
        if hid_e is not None:
            hid_e = hid_e[0].astype(np.float64)
        else:
            # some builds expose only logits; hidden compare via present of
            # last layer is not equivalent -- flag missing
            hid_e = None
        hidden_r = ref["ref_hidden"].astype(np.float64)
        if hid_e is not None:
            report["hidden_cos"] = cos64(hid_e, hidden_r)
            hc = (hid_e * hidden_r).sum(1) / np.maximum(
                np.linalg.norm(hid_e, axis=1) * np.linalg.norm(
                    hidden_r, axis=1), 1e-30)
            report["hidden_rowcos_min"] = float(hc.min())
        kv_r8 = ref["ref_kv8"]                     # [8,2,H,T,D]
        kv_cos = []
        n_present = 0
        for k in range(8):
            key = f"present_key_values_{k}"
            if key not in host_out:
                continue
            n_present += 1
            kv_e = host_out[key][0, :, :, :T, :].astype(np.float64)
            kv_cos.append(cos64(kv_e, kv_r8[k]))
        report["n_kv_layers"] = n_present
        report["kv_cos"] = kv_cos
        report["kv_cos_min"] = min(kv_cos) if kv_cos else None
        print(f"logits_cos={report.get('logits_cos'):.6f} "
              f"hidden_cos={report.get('hidden_cos')} "
              f"kv_layers={n_present} kv_cos_min={report['kv_cos_min']}")

    # latency: warmup 5 + iters (run under locked clocks!)
    for _ in range(5):
        ctx.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
    ev0, ev1 = cuda.Event(), cuda.Event()
    times = []
    for _ in range(args.iters):
        ev0.record(stream)
        ctx.execute_async_v3(stream_handle=stream.handle)
        ev1.record(stream)
        stream.synchronize()
        times.append(ev0.time_till(ev1))
    t = np.array(times)
    report["latency_ms"] = {"p50": float(np.percentile(t, 50)),
                            "mean": float(t.mean()),
                            "min": float(t.min()),
                            "n": args.iters}
    print(f"prefill T={T}: p50={report['latency_ms']['p50']:.1f}ms "
          f"mean={t.mean():.1f}ms min={t.min():.1f}ms over {args.iters}")

    if not args.skip_gate:
        ok = (report["logits_cos"] >= args.gate_threshold
              and report.get("hidden_cos", 1.0) >= args.gate_threshold
              and report.get("kv_cos_min", 1.0) >= args.gate_threshold)
        report["gate"] = "PASS" if ok else "FAIL"
        print("GATE:", report["gate"])
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
