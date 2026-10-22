# Environment

Two hosts: an x86 export host and the Jetson AGX Orin target. This page pins
the certified stack; `scripts/env_*.sh` are the workspace templates.

## x86 export host

| Component | Version | Notes |
|---|---|---|
| Python | 3.10.12 | |
| torch | 2.8.0+cu128 | exporter + vision ONNX export |
| transformers | 5.14.1 | ships `qwen3_5` model code (stock PyPI wheel) |
| onnxruntime | 1.23.2 | x86 ONNX gate (CPUExecutionProvider) |
| CUDA | 12.8 runtime via pip wheels | no system toolkit needed on the export path |

Fresh venv recipe:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.8.0 transformers==5.14.1 onnxruntime==1.23.2 safetensors
```

## Jetson AGX Orin (target)

| Component | certified | Notes |
|---|---|---|
| JetPack / L4T | JP6 / L4T R36.5.0 | A/B comparisons must lock clocks |
| CUDA toolkit (nvcc) | 12.6 | parity-harness + Edge-LLM builds |
| torch | 2.8.0 | runtime gates + parity harness |
| TensorRT | 10.4 wheel | vision engine (trtexec) + text engine runtime |
| TensorRT-Edge-LLM | v0.9.1 + patch bundle | text prefill engine tree |

## Known release gap (TRT 10.4 compat)

TensorRT-Edge-LLM v0.9.1 upstream targets TensorRT >= 10.5; the certified
Orin build carried a small compat delta (trtUtils.cpp, one header, cmake
fragments) so the tree builds against TRT 10.4. Until that delta lands in
`export/patches/`, consumers on TRT 10.4 need it from the certified lineage;
consumers on TRT >= 10.5 build stock v0.9.1 + the GDN patch only.