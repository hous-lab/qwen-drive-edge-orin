#!/usr/bin/env python
"""E1: export the Qwen-Drive vision tower to a static-shape ONNX (v1).

Runs on the x86 4090 box. Loads the full model (bf16, sdpa) once, takes a real
demo scene through the QwenDriveProcessor, then:

  1. records reference outputs of ``vlm.model.visual`` in fp32 / bf16 / fp16
  2. traces a wrapper module (pixel_values -> pooler_output) with the demo
     scene's grid baked in (static shapes, 12 images: 3 views x 4 frames)
  3. writes the ONNX plus an npz bundle (inputs + references) used by the
     x86 onnxruntime check and the Orin gate.

The ONNX carries fp32 weights; TRT applies the fp16 conversion at engine build
so the ONNX stays numerically neutral for gating.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _find_repo():
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

from qwen_drive import QwenDriveForPlanning  # noqa: E402
from qwen_drive.benchmarks import read_scene_file  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=os.environ.get("QD_MODEL_DIR"))
    p.add_argument("--scenes", default=os.environ.get("QD_SCENES"))
    p.add_argument("--image-root", default=os.environ.get("QD_IMAGE_ROOT"))
    p.add_argument("--scene-index", type=int, default=0)
    p.add_argument("--output", default=os.environ.get("QD_ONNX"))
    p.add_argument("--tensors", default=os.environ.get("QD_TENSORS"))
    return p.parse_args()


class VisionWrapper(torch.nn.Module):
    """Static-shape wrapper: pixel_values [N, C*T*P*P] -> merged embeds [N/4, 2560].

    grid_thw is a fixed constant; every derived quantity (bilinear pos-embed
    indices, position ids, cu_seqlens, per-image attention splits) bakes into
    the graph at trace time, which is exactly the static-engine contract.
    """

    def __init__(self, visual, grid_thw):
        super().__init__()
        self.visual = visual
        self.register_buffer("grid_thw", grid_thw, persistent=False)

    def forward(self, pixel_values):
        out = self.visual(pixel_values, grid_thw=self.grid_thw, return_dict=True)
        return out.pooler_output


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    device = "cuda:0"

    model = QwenDriveForPlanning.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    visual = model.vlm.model.visual

    scenes = list(read_scene_file(args.scenes, image_root=args.image_root))
    sample = scenes[args.scene_index]
    inputs = model.processor(sample.scene, device=device)
    pixel_values = inputs["pixel_values"].float()          # [N, C*T*P*P], fp32
    grid = inputs["image_grid_thw"].to(device)             # [num_images, 3]
    num_images, _ = grid.shape
    num_patches, feat = pixel_values.shape
    num_tokens = int(grid.prod(-1).sum().item()) // 4
    print(f"scene={args.scene_index} images={num_images} patches={num_patches} "
          f"tokens={num_tokens} feat={feat}", flush=True)
    print(f"grid={grid.tolist()}", flush=True)

    # ---- references (same module, three dtypes) ----
    ref_bf16 = visual(pixel_values.bfloat16(), grid_thw=grid, return_dict=True).pooler_output
    visual.to(torch.float32)
    ref_fp32 = visual(pixel_values, grid_thw=grid, return_dict=True).pooler_output
    visual.to(torch.float16)
    ref_fp16 = visual(pixel_values.half(), grid_thw=grid, return_dict=True).pooler_output
    for name, ref in [("fp32", ref_fp32), ("bf16", ref_bf16), ("fp16", ref_fp16)]:
        cos = torch.nn.functional.cosine_similarity(
            ref.float().flatten(), ref_fp32.flatten(), dim=0).item()
        print(f"torch {name} vs fp32 cos={cos:.6f}", flush=True)

    # ---- export (fp32 weights; visual currently fp16 -> restore fp32) ----
    visual.to(torch.float32)
    wrapper = VisionWrapper(visual.float(), grid).eval()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (pixel_values,),
        str(out_path),
        opset_version=17,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamo=False,
    )
    print(f"onnx written: {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)", flush=True)

    # ---- test bundle for ORT (x86) and the Orin gate ----
    tpath = Path(args.tensors)
    tpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        tpath,
        pixel_values=pixel_values.cpu().numpy().astype(np.float32),
        grid=grid.cpu().numpy().astype(np.int64),
        ref_fp32=ref_fp32.float().cpu().numpy().astype(np.float32),
        ref_bf16=ref_bf16.float().cpu().numpy().astype(np.float32),
        ref_fp16=ref_fp16.float().cpu().numpy().astype(np.float32),
        scene_index=np.int64(args.scene_index),
        scene_token=np.array(str(sample.scene_token)),
    )
    print(f"tensors written: {tpath}", flush=True)

    manifest = {
        "num_images": num_images, "num_patches": int(num_patches),
        "num_tokens": int(num_tokens), "feat": int(feat),
        "grid": grid.tolist(), "opset": 17,
        "torch_fp32_vs_fp32": 1.0,
        "torch_bf16_vs_fp32_cos": torch.nn.functional.cosine_similarity(
            ref_bf16.float().flatten(), ref_fp32.flatten(), dim=0).item(),
        "torch_fp16_vs_fp32_cos": torch.nn.functional.cosine_similarity(
            ref_fp16.float().flatten(), ref_fp32.flatten(), dim=0).item(),
        "dtype_weights": "fp32",
    }
    with open(out_path.with_suffix(".manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps(manifest, indent=1), flush=True)


if __name__ == "__main__":
    main()
