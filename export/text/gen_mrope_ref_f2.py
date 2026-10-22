#!/usr/bin/env python3
"""True-mrope bf16 reference for the f2 gate (P3-T, 2026-10-06).

Why: text_ref_bundle_f2.npz's reference (ref_logits/ref_kv8) was generated
with an arange 1D rope, and the engine gate fed a matching arange table --
an arange-vs-arange measurement.  mrope_golden_f2.npz (pos3d/rope_table/
kv8_mrope, provenance: inline generator, validated below) is the TRUE mrope
reference for the KV, but has no logits/hidden.

This script runs the bf16 HF model on the f2 inputs_embeds with the golden
pos3d as mrope position_ids [3,1,T], captures logits/hidden/KV8 (pre-rope
k_proj/v_proj hooks, same protocol as eval_int8_scale_fix.py) and:
  1. VALIDATES: v_proj (never roped) vs golden kv8_mrope[*,1] must be
     cos ~= 1.0 per layer -- proves same inputs AND same positions as the
     golden generator.  (k is NOT compared: golden k is POST-rope / cache
     convention -- matches the engine present_key_values, proven by the
     fp16 engine passing 0.9998 vs the bundle -- while our hook is
     pre-rope; pre-vs-post k sits at cos 0.88-0.98 by construction.)
  2. SAVES logs_text/mrope_gate_ref_f2.npz = ref_logits/ref_hidden (ours)
     + ref_kv8 (GOLDEN, post-rope) -- the --ref-npz for
     gate_text_engine_orin.py --rope-npz (golden rope_table).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import os
from safetensors.torch import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

EDG = Path(os.environ.get("QD_RUNTIME_DIR", ".")) / "edgellm"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work", default=str(EDG / "text_work"))
    p.add_argument("--bundle", default=str(EDG / "logs_text" / "text_ref_bundle_f2.npz"))
    p.add_argument("--golden", default=str(EDG / "logs_text" / "mrope_golden_f2.npz"))
    p.add_argument("--out", default=str(EDG / "logs_text" / "mrope_ref_logits_f2.npz"))
    p.add_argument("--out-report", default=str(EDG / "logs_text" / "mrope_ref_logits_f2.json"))
    return p.parse_args()


def load_bf16_model(work):
    cfg = AutoConfig.from_pretrained(work)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval().cuda()
    prefix = "vlm.model.language_model."
    sd = {}
    with safe_open(str(Path(work, "model.safetensors").resolve()),
                   framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(prefix):
                sd["model." + k[len(prefix):]] = f.get_tensor(k)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    assert not missing, f"missing after remap: {missing[:8]}"
    assert not unexpected, f"unexpected after remap: {unexpected[:8]}"
    return model


class KVHook:
    def __init__(self):
        self.out = None

    def __call__(self, mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        self.out = t.detach().float().cpu()


def cos64(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    args = parse_args()
    torch.set_grad_enabled(False)

    model = load_bf16_model(args.work)
    cfg = model.config
    full_attn = [i for i, t in enumerate(cfg.layer_types) if t != "linear_attention"]
    print(f"[gen] bf16 model loaded; full-attn layers: {full_attn}")

    z = np.load(args.bundle)
    g = np.load(args.golden)
    emb = torch.from_numpy(z["inputs_embeds_fp32"]).bfloat16().cuda().unsqueeze(0)
    pos3d = torch.from_numpy(g["pos3d"]).long().cuda()      # [3,T]
    T = emb.shape[1]
    print(f"[gen] f2 T={T}; pos3d {tuple(pos3d.shape)}")

    hooks, handles = [], []
    for i in full_attn:
        for proj in ("k_proj", "v_proj"):
            h = KVHook()
            handles.append(model.get_submodule(
                f"model.layers.{i}.self_attn.{proj}").register_forward_hook(h))
            hooks.append(((i, proj), h))

    out = model(inputs_embeds=emb, position_ids=pos3d[:, None, :],
                use_cache=False, output_hidden_states=True)
    logits = out.logits[0, -1].float().cpu().numpy()
    hidden = out.hidden_states[-1][0].float().cpu().numpy()

    # KV in bundle layout [8, 2, H, T, D] (0=k, 1=v)
    kv8 = None
    for (i, proj), h in hooks:
        t = h.out.numpy()[0]                                # [T, H*D]
        H = 4
        D = t.shape[1] // H
        arr = t.reshape(t.shape[0], H, D).transpose(1, 0, 2)  # [H, T, D]
        if kv8 is None:
            kv8 = np.zeros((len(full_attn), 2, H, T, D), np.float32)
        kv8[full_attn.index(i), 0 if proj == "k_proj" else 1] = arr
    print(f"[gen] kv8 {kv8.shape}")

    kv_g = g["kv8_mrope"]
    assert kv_g.shape == kv8.shape, (kv_g.shape, kv8.shape)
    per = {}
    for li, i in enumerate(full_attn):
        # v only (never roped): positions + input equivalence check
        per[f"L{i}"] = round(cos64(kv8[li, 1], kv_g[li, 1]), 8)
    kv_min = min(per.values())
    print(f"[gen] VALIDATE v-vs-golden (rope-independent): min={kv_min:.8f} "
          f"per={per}")

    logits_bundle = cos64(logits, z["ref_logits"])
    print(f"[gen] logits cos(mrope-ref, arange-bundle) = {logits_bundle:.6f} "
          f"(true mrope effect on last-token logits)")

    # golden rope_table rows == HF rotary(cos||sin) at pos3d?  (vision rows
    # exercise the interleaved mrope frequency order -- a 1D-degenerate
    # check cannot distinguish interleave conventions)
    rot = model.model.rotary_emb
    cos_hf, _sin_hf = rot(emb, pos3d[:, None, :])
    rt = g["rope_table"][0, :T].astype(np.float64)          # [T,64]
    ch = cos_hf[0].float().cpu().numpy().astype(np.float64)  # [T,64]
    row_err = float(np.abs(rt - ch).max())
    print(f"[gen] golden rope_table vs HF rotary rows: max|diff|={row_err:.3e}"
          f" (vision-row max="
          f"{float(np.abs(rt[(pos3d[1].cpu() != pos3d[0].cpu()) | (pos3d[2].cpu() != pos3d[0].cpu())] - ch[(pos3d[1].cpu() != pos3d[0].cpu()) | (pos3d[2].cpu() != pos3d[0].cpu())]).max()):.3e})")

    # gate ref = our logits/hidden + GOLDEN kv8 (post-rope, engine convention)
    gate_ref = str(Path(args.out).parent / "mrope_gate_ref_f2.npz")
    np.savez(str(gate_ref), ref_logits=logits, ref_hidden=hidden,
             ref_kv8=kv_g)
    np.savez(str(args.out), ref_logits=logits, ref_hidden=hidden, ref_kv8=kv8)
    report = {"T": int(T), "v_vs_golden_min": kv_min, "v_vs_golden_per": per,
              "logits_cos_mrope_vs_arange_bundle": round(logits_bundle, 6),
              "gate_ref": gate_ref,
              "kv_convention": "golden kv8_mrope k is POST-rope (engine "
                               "cache convention); ours pre-rope, v identical",
              "pos3d_source": args.golden, "bundle": args.bundle}
    Path(args.out_report).write_text(json.dumps(report, indent=2) + "\n")
    print(f"[gen] wrote {args.out} + gate ref {gate_ref} + {args.out_report}")
    assert kv_min > 0.999, "our mrope forward does NOT reproduce the golden v"
    print("[gen] OK")


if __name__ == "__main__":
    main()
