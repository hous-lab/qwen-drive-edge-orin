# ckpts — artifact policy and download table

**Upstream-hosted files are LINK-ONLY by policy.** This repo does not ship or
rehost any third-party weights or dataset files — and the same applies to
retrained/QAT variants derived from them. Download upstream files from the
sources below into the workspace layout, then verify sha256 before running
the pipeline.

## Upstream artifacts (link-only)

| Artifact | Source | License | Local placement |
|---|---|---|---|
| Qwen-Drive-1.0-4B weights | https://huggingface.co/Qwen/Qwen-Drive-1.0-4B | Apache-2.0 | `$QD_MODEL_DIR` (root: config.json, *.safetensors, tokenizer) |
| Qwen-Drive-1.0-4B planner weights | same HF repo, `planner-rl/` subfolder (the certified gate planner) | Apache-2.0 | `$QD_MODEL_DIR/planner-rl` |
| Upstream inference code | https://github.com/QwenLM/qwen-drive | Apache-2.0 | `$QD_UPSTREAM_REPO` (`pip install -e`) |
| TensorRT-Edge-LLM v0.9.1 | https://github.com/NVIDIA/TensorRT-Edge-LLM (tag v0.9.1) | Apache-2.0 | `$QD_EDGELLM_SRC` |
| PhysicalAI Autonomous Vehicles chunk 0 | https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles | NVIDIA AV Dataset License (gated) | `$QD_DATA_ROOT` per [eval/adapter/pai_to_qwen.py](../eval/adapter/pai_to_qwen.py) |
| AlpaSim closed-loop env | https://github.com/NVlabs/alpasim | — | external env; closed-loop is documented as a pointer only |
| NVlabs alpamayo-recipes | https://github.com/NVlabs/alpamayo-recipes | Apache-2.0 | `$QD_ALPAMAYO_RECIPES` (PAI dataset interface for [eval/adapter/](../eval/adapter/)) |

sha256 pins: the upstream HF/GitHub hosts publish per-file hashes; verify
against those. The certified production run used the checkpoint exactly as
published upstream on 2026-10-03..12 (no fine-tuning or modification).

## Self-produced artifacts (this project's outputs)

Self-produced artifacts are published on the hous-lab Hugging Face org
(hf.co/hous-lab/qwen-drive-edge-orin) and referenced by sha256:

```bash
hf download hous-lab/qwen-drive-edge-orin --local-dir "$QD_RUNTIME_DIR/edgellm/logs_text"
```

| Artifact | What it is | Size | sha256 (first 8) | Provenance |
|---|---|---|---|---|
| mrope_golden_f2.npz | Composite M-RoPE reference for the f2 manifest: pos3d, rope_table rows, post-rope golden KV of the full-attention layers | ~76 MB | `0f4e8bad` | development-time reference generator over the upstream checkpoint (validated by [export/text/gen_mrope_ref_f2.py](../export/text/gen_mrope_ref_f2.py)) |
| text_ref_bundle_f2.npz | f2 gate inputs: inputs_embeds + arange-rope reference logits/KV | ~105 MB | `795d7bf2` | chain reference dumps over the upstream checkpoint |

Both are derived exclusively from the upstream checkpoint (link-only) plus
self-written generation code, so they are self-produced by policy. The full
sha256 of the exact published files is recorded on the HF repo card.
[export/text/gen_mrope_ref_f2.py](../export/text/gen_mrope_ref_f2.py) consumes both, validates the v-KV against
the golden (cos > 0.999), and writes the gate reference the text-engine gate
compares against.
