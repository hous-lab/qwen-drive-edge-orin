# Environment

Two hosts: an x86 export host and the Jetson AGX Orin target. This page pins the verified stack; `scripts/env_*.sh` are the workspace templates.

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

| Component | Version | Notes |
|---|---|---|
| JetPack / L4T | JP6 / L4T R36.5.0 | A/B comparisons must lock clocks |
| CUDA toolkit (nvcc) | 12.6 | parity-harness + Edge-LLM builds |
| torch | 2.8.0 | runtime gates + parity harness |
| TensorRT | 10.4 wheel | vision engine (trtexec) + text engine runtime |
| TensorRT-Edge-LLM | v0.9.1 + patch bundle | text prefill engine tree |

## Known release gap (TRT 10.4 compat)

TensorRT-Edge-LLM v0.9.1 upstream targets TensorRT >= 10.5; the production Orin build carried a compat delta so the tree builds against TRT 10.4. That delta is now shipped in [export/patches/orin_text_engine/](../export/patches/orin_text_engine/) as eight `files/` overrides (trtUtils.cpp, cuda126_cutedsl_compat.h, the top-level and cpp/ CMakeLists.txt, CuteDsl.cmake, XQACubins.cmake, fusedNvfp4GemmAllReducePlugin.cpp, contextFMHARunner.cpp) and is staged automatically by [build_plugin_orin.sh](../export/patches/orin_text_engine/build_plugin_orin.sh); consumers on TRT >= 10.5 build fine from it too (the overrides are no-ops there) or from stock v0.9.1 + the GDN patch only.
