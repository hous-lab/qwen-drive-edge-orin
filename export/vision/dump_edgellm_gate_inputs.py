#!/usr/bin/env python3
"""Dump the 5 Edge-LLM visual engine inputs + reference outputs for Orin gating."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_edgellm_onnx_x86 import build_aux_inputs  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tensors", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    z = np.load(a.tensors)
    pv, grid = z["pixel_values"], z["grid"]
    rotary, cu, idx, w = build_aux_inputs(grid)
    np.savez(a.out,
             input=pv.astype(np.float16),
             rotary_pos_emb=rotary,
             cu_seqlens=cu,
             fast_pos_embed_idx=idx,
             fast_pos_embed_weight=w,
             grid=grid,
             ref_fp32=z["ref_fp32"], ref_bf16=z["ref_bf16"], ref_fp16=z["ref_fp16"],
             scene_token=z["scene_token"])
    print(f"wrote {a.out}: T={pv.shape[0]}, cu={cu.tolist()}")


if __name__ == "__main__":
    main()
