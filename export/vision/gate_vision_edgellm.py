#!/usr/bin/env python3
"""Gate + latency for the Edge-LLM qwen3.5 visual engine on Orin.

5-input interface: input[T,1536] fp16, rotary_pos_emb[T,32] fp32,
cu_seqlens[N+1] int32, fast_pos_embed_idx[4,T] int64,
fast_pos_embed_weight[4,T] fp16 -> output[T,2560] fp16.
Requires the edgellm plugin .so; compares against bundle refs (fp32/bf16/fp16).
"""
import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import pycuda.autoinit  # noqa: F401  MUST precede all TRT usage
import pycuda.driver as cuda
import tensorrt as trt
import os

PLUGIN_SO = os.environ["QD_PLUGIN_SO"]  # Edge-LLM plugin .so (see kernels/README.md)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True)
    p.add_argument("--inputs", required=True, help="npz from dump_edgellm_gate_inputs.py")
    p.add_argument("--tag", default="demo")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--out-dir", default=os.environ.get("QD_RUNTIME_DIR", "."))
    p.add_argument("--save-out", action="store_true")
    return p.parse_args()


def cos64(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def row_cos(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.maximum(den, 1e-30)


def main():
    args = parse_args()
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    lib = ctypes.CDLL(PLUGIN_SO, mode=ctypes.RTLD_GLOBAL)
    lib.initEdgellmPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.initEdgellmPlugins.restype = ctypes.c_bool
    assert lib.initEdgellmPlugins(None, b""), "initEdgellmPlugins failed"

    with open(args.engine, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    print("engine IO:")
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        print(f"  {n}: {engine.get_tensor_dtype(n)} "
              f"{engine.get_tensor_shape(n)} "
              f"{'IN' if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT else 'OUT'}")

    z = np.load(args.inputs)
    cu32 = z["cu_seqlens"].astype(np.int32)
    max_seq = int(np.diff(cu32).max())
    # vitAttentionPlugin contract: shape [max_seqlen] carries the value; contents ignored
    feeds = {
        "input": z["input"].astype(np.float16),
        "rotary_pos_emb": z["rotary_pos_emb"].astype(np.float32),
        "cu_seqlens": cu32,
        "fast_pos_embed_idx": z["fast_pos_embed_idx"].astype(np.int64),
        "fast_pos_embed_weight": z["fast_pos_embed_weight"].astype(np.float16),
        "max_seqlen_carrier": np.zeros(max_seq, dtype=np.int32),
    }
    print(f"max_seqlen_carrier shape=({max_seq},)")
    T = feeds["input"].shape[0]
    stream = cuda.Stream()

    # set shapes + allocate per engine-resolved names
    dev_in, host_out = {}, {}
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(n)
        if mode == trt.TensorIOMode.INPUT:
            base = n.split("[")[0].strip()
            arr = None
            for key, val in feeds.items():
                if key in n or n in key or base == key:
                    arr = val
                    break
            assert arr is not None, f"no feed for engine input {n}"
            ctx.set_input_shape(n, arr.shape)
            d = cuda.mem_alloc(arr.nbytes)
            dev_in[n] = (arr, d)
        else:
            shape = tuple(ctx.get_tensor_shape(n))
            dt = trt.nptype(engine.get_tensor_dtype(n))
            host_out[n] = np.empty(shape, dtype=dt)
            dev_in[n] = (host_out[n], cuda.mem_alloc(int(np.prod(shape)) * np.dtype(dt).itemsize))

    for n, (arr, d) in dev_in.items():
        ctx.set_tensor_address(n, int(d))
        if n in feeds:
            cuda.memcpy_htod_async(d, arr, stream)

    ctx.execute_async_v3(stream_handle=stream.handle)
    for n in host_out:
        cuda.memcpy_dtoh_async(host_out[n], int(dev_in[n][1]), stream)
    stream.synchronize()

    out = list(host_out.values())[0]
    print(f"output {out.shape} {out.dtype}, T={T}")
    report = {"tag": args.tag, "engine": Path(args.engine).name, "T": int(T)}
    for ref_name in ("ref_fp32", "ref_bf16", "ref_fp16"):
        ref = z[ref_name]
        rc = row_cos(out, ref)
        report[ref_name] = {
            "cos": cos64(out, ref),
            "rowcos_p1": float(np.percentile(rc, 1)),
            "rowcos_p50": float(np.percentile(rc, 50)),
            "n_below_0.999": int((rc < 0.999).sum()),
            "n_below_0.99": int((rc < 0.99).sum()),
        }
        print(f"{ref_name}: cos={report[ref_name]['cos']:.6f} "
              f"rowcos p1={report[ref_name]['rowcos_p1']:.5f} "
              f"n<0.999={report[ref_name]['n_below_0.999']} "
              f"n<0.99={report[ref_name]['n_below_0.99']}")

    # latency: warmup + iters
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
    print(f"latency p50={report['latency_ms']['p50']:.1f}ms "
          f"mean={t.mean():.1f}ms min={t.min():.1f}ms over {args.iters}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"gate_edgellm_engine_{args.tag}.json").write_text(json.dumps(report, indent=2))
    if args.save_out:
        np.savez(out_dir / f"edgellm_engine_out_{args.tag}.npz", output=out.astype(np.float32))
    gate = report["ref_fp32"]["cos"]
    print("GATE vs ref_fp32:", "PASS" if gate >= 0.999 else "FAIL", f"({gate:.6f})")


if __name__ == "__main__":
    main()

