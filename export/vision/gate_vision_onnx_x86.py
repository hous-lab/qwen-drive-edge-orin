#!/usr/bin/env python
"""E1 x86 gate: onnxruntime (CPU) vs torch references in the tensor bundle.

Usage: gate_vision_onnx_x86.py --onnx vision_fp32.onnx --tensors vision_test_tensors.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def cos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--tensors", required=True)
    p.add_argument("--output", default=None, help="optional json report path")
    args = p.parse_args()

    bundle = np.load(args.tensors, allow_pickle=True)
    pv = bundle["pixel_values"]
    refs = {k: bundle[f"ref_{k}"] for k in ("fp32", "bf16", "fp16")}

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    (ort_out,) = sess.run(None, {"pixel_values": pv})

    report = {"ort_shape": list(ort_out.shape)}
    for name, ref in refs.items():
        report[f"ort_vs_{name}_cos"] = cos(ort_out, ref)
        report[f"ort_vs_{name}_maxabs"] = float(np.abs(ort_out - ref).max())
    print(json.dumps(report, indent=1))

    gate = report["ort_vs_fp32_cos"]
    print(f"GATE ort-vs-fp32 cos={gate:.6f} -> {'PASS' if gate >= 0.999 else 'FAIL'}")
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
