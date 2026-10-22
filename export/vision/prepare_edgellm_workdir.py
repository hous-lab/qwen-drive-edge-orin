#!/usr/bin/env python3
"""Build a qwen3_5-form work copy of Qwen-Drive for TensorRT-Edge-LLM export.

The work copy must satisfy Edge-LLM's qwen3_5 export contract:
  - config.json rewritten to standard qwen3_5 form
    (root model_type="qwen3_5", root-level vision_config, nested text_config)
  - visual weights remapped  vlm.model.visual.*  ->  model.visual.*
    (the generic qwen3_5 path in export.py has no in-memory remap; only the
     alpamayo branch has _prepare_alpamayo_visual_params)

Only the 297 visual tensors are copied (visual-only export); bf16 preserved,
the exporter casts to fp16 itself.  LLM tensors are intentionally omitted.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import os
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir",
                   default=os.environ.get("QD_MODEL_DIR"))
    p.add_argument("--work-dir",
                   default=os.environ.get("QD_WORK_DIR"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_work_config(cfg: dict) -> dict:
    vlm = cfg["vlm_config"]
    vis = dict(vlm["vision_config"])
    # strip HF AutoConfig noise keys so the work copy stays minimal
    for k in ("_name_or_path", "architectures", "id2label", "label2id",
              "chunk_size_feed_forward", "problem_type", "return_dict",
              "output_attentions", "output_hidden_states", "is_encoder_decoder",
              "initializer_range", "dtype"):
        vis.pop(k, None)
    txt = dict(vlm["text_config"])
    for k in ("_name_or_path", "architectures", "id2label", "label2id",
              "chunk_size_feed_forward", "problem_type", "return_dict",
              "output_attentions", "output_hidden_states",
              "is_encoder_decoder"):
        txt.pop(k, None)
    work = {
        "model_type": "qwen3_5",
        "vision_config": vis,
        "text_config": txt,
    }
    return work


def remap_visual_weights(model_dir: Path) -> dict:
    """Stream visual tensors out of the checkpoint with prefix remap."""
    src = model_dir / "model.safetensors"
    out = {}
    with safe_open(src, framework="pt") as f:
        keys = [k for k in f.keys() if k.startswith("vlm.model.visual.")]
        for k in keys:
            out["model.visual." + k[len("vlm.model.visual."):]] = f.get_tensor(k)
    return out


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    work_dir = Path(args.work_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    work_cfg = build_work_config(cfg)

    with safe_open(model_dir / "model.safetensors", framework="pt") as f:
        n_visual = sum(1 for k in f.keys() if k.startswith("vlm.model.visual."))
    print(f"source: {n_visual} visual tensors under vlm.model.visual.*")
    if args.dry_run:
        print("work config.json would be:")
        print(json.dumps(work_cfg, indent=2)[:2000])
        return

    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "config.json").write_text(json.dumps(work_cfg, indent=2) + "\n")
    print(f"wrote {work_dir / 'config.json'}")

    weights = remap_visual_weights(model_dir)
    n_bytes = sum(t.numel() * t.element_size() for t in weights.values())
    save_file(weights, str(work_dir / "model.safetensors"))
    print(f"wrote {work_dir / 'model.safetensors'}: "
          f"{len(weights)} tensors, {n_bytes / 2**30:.2f} GiB")
    # self-check: reload and verify key prefix + one tensor shape
    with safe_open(work_dir / "model.safetensors", framework="pt") as f:
        ks = list(f.keys())
        assert all(k.startswith("model.visual.") for k in ks), "bad prefix"
        assert len(ks) == n_visual, f"count mismatch {len(ks)} vs {n_visual}"
        t = f.get_tensor("model.visual.pos_embed.weight")
        print(f"self-check OK: pos_embed {tuple(t.shape)} {t.dtype}")


if __name__ == "__main__":
    main()
