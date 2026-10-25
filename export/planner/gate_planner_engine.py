#!/usr/bin/env python3
"""Gate + latency for the planner one-step TRT engine (Jetson Orin).

Gate: engine endpoint vs production-bf16 reference (bundle["ref_prod"], the
official expert) — cos >= 0.999, plus the wrapper16 twin for continuity with
the export gate B. Inputs are byte-exact production-rounded values from
dump_gate_bundle.py (bf16-quantized, upcast to the engine's fp32 I/O here).

Latency: inputs are uploaded ONCE and stay resident (production scene KV comes
from the on-device prefill cache; the ~220MB host feed is setup, not per-call).
Timing is cuda-event p50 over N replays on the current stream, locked clocks.

Usage (Orin, planner venv):
  python gate_planner_engine.py --plan engines/planner/planner_onestep_s3385_bf16.plan \
      --bundle planner_gate_bundle.npz [--iters 200] [--warmup 20]
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tensorrt as trt


def cos_sim(a, b):
    a, b = a.double().ravel(), b.double().ravel()
    return float(a @ b / (a.norm() * b.norm()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--scene-len", type=int, default=None,
                    help="S for dynamic-shape engines (bundle must match)")
    ap.add_argument("--tag", default="bf16", help="engine precision tag for the report")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    bundle = np.load(args.bundle)
    ref_prod = torch.from_numpy(bundle["ref_prod"])
    wrapper16 = torch.from_numpy(bundle["wrapper16_out"])

    # ---- engine setup ----
    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.plan, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine is None:
        sys.exit("engine deserialization failed")
    context = engine.create_execution_context()
    print(f"plan={args.plan} trt={trt.__version__} "
          f"device={torch.cuda.get_device_name(0)}", flush=True)
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        print(f"  {'IN ' if mode == trt.TensorIOMode.INPUT else 'OUT'} {name} "
              f"{shape} {np.dtype(dtype).name}", flush=True)

    # ---- device inputs: upcast production bf16 values to the engine's fp32 I/O ----
    def dev_input(name):
        arr = bundle[name]
        t = torch.from_numpy(np.ascontiguousarray(arr))
        if t.dtype == torch.bfloat16:
            t = t.float()  # exact upcast of production values
        return t.to("cuda:0")

    in_names = ["waypoints", "flow_time", "hq_pose", "hq_velocity", "hq_accel",
                "scene_k", "scene_v", "nav_command", "ego_status", "cos", "sin"]
    dev = {n: dev_input(n) for n in in_names}
    out_name = next(engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                    if engine.get_tensor_mode(engine.get_tensor_name(i))
                    == trt.TensorIOMode.OUTPUT)

    # dynamic-shape engine: bind scene_k/v at the requested S per invocation
    dyn = engine.get_tensor_shape("scene_k")[1] < 0
    if dyn:
        assert args.scene_len, "--scene-len required for a dynamic engine"
        lo, opt, hi = engine.get_tensor_profile_shape("scene_k", 0)
        assert lo[1] <= args.scene_len <= hi[1], \
            f"S={args.scene_len} outside profile [{lo[1]},{hi[1]}]"
        for name in ("scene_k", "scene_v"):
            assert tuple(dev[name].shape)[1] == args.scene_len, \
                f"bundle {name} S={tuple(dev[name].shape)[1]} != --scene-len"
            assert context.set_input_shape(name, tuple(dev[name].shape)), name
        print(f"[dyn] profile S[{lo[1]}..{hi[1]}] opt={opt[1]}; gating at "
              f"S={args.scene_len}", flush=True)

    out_buf = torch.empty(tuple(engine.get_tensor_shape(out_name)),
                          dtype=torch.float32, device="cuda:0")

    for n, t in dev.items():
        context.set_tensor_address(n, t.data_ptr())
    context.set_tensor_address(out_name, out_buf.data_ptr())
    # dedicated non-default stream: enqueueV3 on the legacy default stream makes
    # TRT insert cudaStreamSynchronize per call (measured ~2x latency noise)
    stream = torch.cuda.Stream()

    def run():
        with torch.cuda.stream(stream):
            context.execute_async_v3(stream.cuda_stream)

    # ---- correctness ----
    run()
    torch.cuda.synchronize()
    got = out_buf[0].cpu()  # [50, 3] fp32
    cos_prod = cos_sim(got, ref_prod[0])
    cos_wrap = cos_sim(got, wrapper16[0])
    d_prod = (got - ref_prod[0]).abs().max().item()
    ok = cos_prod >= 0.999
    print(f"[gate] TRT({args.tag}) vs ref_prod: cos={cos_prod:.8f} "
          f"max|d|={d_prod*100:.2f}cm -> {'PASS' if ok else 'FAIL'}", flush=True)
    print(f"[gate] TRT({args.tag}) vs wrapper16: cos={cos_wrap:.8f}", flush=True)

    # ---- latency (locked clocks; engine execution only, inputs resident) ----
    # events MUST be recorded on the SAME stream as the work — recording on the
    # default stream while executing on a dedicated stream times nothing
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    times = []
    for _ in range(args.iters):
        with torch.cuda.stream(stream):
            start.record(stream)
            context.execute_async_v3(stream.cuda_stream)
            stop.record(stream)
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop))
    p50 = statistics.median(times)
    print(f"[latency] TRT({args.tag}) engine p50 {p50:.2f} ms  min {min(times):.2f}  "
          f"max {max(times):.2f}  (n={args.iters}, inputs resident)", flush=True)
    print("GATE", "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
