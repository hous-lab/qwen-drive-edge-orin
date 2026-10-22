#!/usr/bin/env python3
"""Gate the official TensorRT-Edge-LLM visual ONNX against torch references.

Builds the 5 dynamic inputs (input / rotary_pos_emb / cu_seqlens /
fast_pos_embed_idx / fast_pos_embed_weight) from grid_thw using the same
transformers.vision_utils helpers the HF forward uses, runs ORT CPU, and
compares against the reference bundle (ref_fp32 / ref_bf16 / ref_fp16).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import os
from transformers.vision_utils import (
    get_vision_bilinear_indices_and_weights,
    get_vision_cu_seqlens,
    get_vision_position_ids,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionRotaryEmbedding


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx",
                   default=os.environ.get("QD_ONNX"))
    p.add_argument("--tensors",
                   default=os.environ.get("QD_TENSORS"))
    p.add_argument("--tag", default="edgellm_demo")
    p.add_argument("--out-dir",
                   default=os.environ.get("QD_RUNTIME_DIR", "."))
    return p.parse_args()


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.maximum(den, 1e-30)


def build_aux_inputs(grid: np.ndarray):
    """grid_thw [N,3] -> (rotary[T,32] fp32, cu int32, idx [4,T] i64, w [4,T] fp16)."""
    grid_t = torch.from_numpy(grid)
    pos_ids = get_vision_position_ids(grid_t, 2)          # [T,2]
    rot = Qwen3_5VisionRotaryEmbedding(32)                # head_dim 64 // 2
    rotary = rot(pos_ids)                                 # [T,32] fp32
    cu = get_vision_cu_seqlens(grid_t)                    # [N+1] int32
    idx, w = get_vision_bilinear_indices_and_weights(grid_t, 48, 2)
    return (rotary.numpy().astype(np.float32),
            cu.numpy().astype(np.int32),
            idx.numpy().astype(np.int64),
            w.numpy().astype(np.float16))


def main():
    args = parse_args()
    z = np.load(args.tensors)
    pv, grid = z["pixel_values"], z["grid"]
    rotary, cu, idx, w = build_aux_inputs(grid)
    T = pv.shape[0]
    assert idx.shape[1] == T, f"idx {idx.shape} vs T={T}"
    print(f"T={T} imgs={len(grid)} cu={cu.tolist()}")

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    print("onnx inputs:", [(i.name, i.type, i.shape) for i in sess.get_inputs()])
    print("onnx outputs:", [(o.name, o.type, o.shape) for o in sess.get_outputs()])
    feed = {
        "input": pv.astype(np.float16),
        "rotary_pos_emb": rotary,
        "cu_seqlens": cu,
        "fast_pos_embed_idx": idx,
        "fast_pos_embed_weight": w,
    }
    feed = {n: feed[n] for n in names}
    out = sess.run(None, feed)[0]
    print(f"output {out.shape} {out.dtype}")

    report = {"tag": args.tag, "T": int(T), "n_images": int(len(grid))}
    for ref_name in ("ref_fp32", "ref_bf16", "ref_fp16"):
        ref = z[ref_name]
        assert ref.shape[0] == out.shape[0]
        rc = row_cos(out, ref)
        report[ref_name] = {
            "cos": cos(out, ref),
            "rowcos_p1": float(np.percentile(rc, 1)),
            "rowcos_p5": float(np.percentile(rc, 5)),
            "rowcos_p50": float(np.percentile(rc, 50)),
            "n_below_0.999": int((rc < 0.999).sum()),
            "n_below_0.99": int((rc < 0.99).sum()),
        }
        print(f"{ref_name}: cos={report[ref_name]['cos']:.6f} "
              f"rowcos p1={report[ref_name]['rowcos_p1']:.5f} "
              f"n<0.999={report[ref_name]['n_below_0.999']}")

    out_path = Path(args.out_dir) / f"gate_edgellm_{args.tag}.json"
    out_path.write_text(json.dumps(report, indent=2))
    np.savez(Path(args.out_dir) / f"edgellm_out_{args.tag}.npz",
             output=out.astype(np.float32))
    print(f"wrote {out_path}")
    gate = report["ref_fp32"]["cos"]
    print("GATE vs ref_fp32:", "PASS" if gate >= 0.999 else "FAIL", f"({gate:.6f})")


if __name__ == "__main__":
    main()
