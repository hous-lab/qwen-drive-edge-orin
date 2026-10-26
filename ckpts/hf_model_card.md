---
license: apache-2.0
tags:
- tensorrt
- jetson
- orin
- vla
- autonomous-driving
- edge-deployment
---

# qwen-drive-edge-orin — gate reference artifacts

Reference bundles for the text-engine gate of the [qwen-drive-edge-orin](https://github.com/hous-lab/qwen-drive-edge-orin) pipeline (Qwen-Drive-1.0 4B VLA on Jetson AGX Orin, self-developed sm_87 GDN chunk-prefill kernel in NVIDIA TensorRT-Edge-LLM). These files are gate INPUTS: the pipeline validates them and derives the final gate reference from them at run time.

## Files

| File | Size | What it is |
|---|---|---|
| `mrope_golden_f2.npz` | ~76 MB | Composite M-RoPE reference for the f2 manifest: pos3d, rope_table rows, post-rope golden KV of the 8 full-attention layers |
| `text_ref_bundle_f2.npz` | ~105 MB | f2 gate inputs: inputs_embeds [T=1141, 2560] fp16 + arange-rope reference logits/KV |

sha256 pins (first 8 hex) for both files are in [ckpts/README.md](https://github.com/hous-lab/qwen-drive-edge-orin/blob/main/ckpts/README.md) of the GitHub repo. Full hashes of published files:

- `text_ref_bundle_f2.npz`: `795d7bf23267abf537623667e26deadfec2ed273f5b7490f5c6628dfed6e386a` (verified against the HF LFS hash at upload)
- `mrope_golden_f2.npz`: `0f4e8bad89d74da807f50f4a033117794783d522362763186afd5cdf9acbb76e` (verified against the HF LFS hash at upload)

## Usage

```bash
hf download hous-lab/qwen-drive-edge-orin --local-dir "$QD_RUNTIME_DIR/edgellm/logs_text"
# validate + derive the gate reference (x86, torch bf16 forward):
$QD_PY $REPO/export/text/gen_mrope_ref_f2.py
# Orin text-engine gate consuming the derived reference:
EDGELLM_PLUGIN_PATH=$QD_PLUGIN_SO $QD_PY export/text/gate_text_engine_orin.py \
  --rope-npz mrope_golden_f2.npz --ref-npz mrope_gate_ref_f2.npz
```

## Provenance and attribution

Derived exclusively from the upstream checkpoint [Qwen/Qwen-Drive-1.0-4B](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B) (Apache-2.0) plus self-written generation code. **Upstream weights are NOT rehosted here.** Generator scripts and the full pipeline live in the GitHub repo linked above. Eval scenes trace to the gated NVIDIA PhysicalAI AV dataset (chunk 0); no dataset content is redistributed here either.

## Platform pairing

Consumed on x86 (torch 2.8.0 / transformers 5.14.1) and deployed on Jetson AGX Orin (sm_87, TRT 10.4, JetPack 6.2.2 / L4T R36.5.0). The KV tensors are fp16 engine-contract tensors sized to the f2 manifest (T=1141); other shapes need regeneration, not slicing.
