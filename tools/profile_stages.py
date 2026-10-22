#!/usr/bin/env python3
"""L2 stage breakdown with CUDA events: vision / text-prefill / planner (+ layer types)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import os

def _find_repo():
    """Locate the upstream qwen-drive repo via QD_UPSTREAM_REPO (or walk-up)."""
    env = os.environ.get("QD_UPSTREAM_REPO")
    for cand in [*([Path(env)] if env else []), *Path(__file__).resolve().parents]:
        if (cand / "src" / "qwen_drive" / "__init__.py").exists():
            return cand
    raise RuntimeError(
        "qwen_drive package not found: set QD_UPSTREAM_REPO (see ckpts/README.md)")
sys.path.insert(0, str(REPO / "src"))
from qwen_drive import InferenceMode, QwenDriveForPlanning  # noqa: E402
from qwen_drive.benchmarks import read_scene_file  # noqa: E402
from qwen_drive.images import ImageArchive  # noqa: E402


class StageTimer:
    """CUDA-event timer around a module forward."""

    def __init__(self):
        self.records = []

    def wrap(self, module, name):
        orig = module.forward

        def timed(*a, **k):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = orig(*a, **k)
            e.record()
            self.records.append((name, s, e))
            return out

        module.forward = timed
        return module

    def collect(self):
        torch.cuda.synchronize()
        out = {}
        for name, s, e in self.records:
            out.setdefault(name, []).append(e.elapsed_time(s) * -1.0)
        self.records = []
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--planner", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--image-archive", default=None)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timed", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    device = "cuda:0"

    model = QwenDriveForPlanning.from_pretrained(
        args.model_dir, planner=args.planner,
        dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()

    archive = ImageArchive.open(args.image_archive) if args.image_archive else None
    scenes = list(read_scene_file(args.scenes, image_root=args.image_root, image_archive=archive))

    timer = StageTimer()
    timer.wrap(model.vlm.model.visual, "vision_tower")
    timer.wrap(model.vlm.model.language_model, "text_prefill")
    timer.wrap(model.planning_expert, "planner_total")

    # per-layer-type: one DeltaNet layer (0) and one full-attention layer (3)
    lm = model.vlm.model.language_model
    layers = getattr(getattr(lm, "model", lm), "layers", None)
    if layers is None:
        layers = lm.layers
    timer.wrap(layers[0], "layer_delta_0")
    timer.wrap(layers[3], "layer_full_3")
    timer.wrap(layers[6], "layer_delta_6")

    # planner per-step via predict_endpoint
    orig_ep = model.planning_expert.predict_endpoint

    def timed_ep(*a, **k):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = orig_ep(*a, **k)
        e.record()
        timer.records.append(("planner_step", s, e))
        return out

    model.planning_expert.predict_endpoint = timed_ep

    totals = {}
    for i, sample in enumerate(scenes):
        inputs = model.processor(sample.scene, with_reasoning=False, device="cpu")
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        timer.records = []
        t0 = time.perf_counter()
        model.plan_from_inputs(inputs, mode=InferenceMode.DIRECT_PLANNING,
                               num_steps=args.num_steps, seed=42)
        wall = (time.perf_counter() - t0) * 1e3
        got = timer.collect()
        for name, vals in got.items():
            totals.setdefault(name, []).extend(vals)
        totals.setdefault("wall", []).append(wall)

    report = {}
    for name, vals in totals.items():
        arr = np.array(vals[1:]) if name != "wall" else np.array(vals)  # drop warmup scene? keep simple: report all
        report[name] = {"n": len(vals), "mean_ms": float(np.mean(vals)), "p50_ms": float(np.median(vals))}
    report["_config"] = {"scenes": len(scenes), "num_steps": args.num_steps,
                         "gpu": torch.cuda.get_device_name(0)}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

