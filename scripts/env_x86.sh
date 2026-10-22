#!/bin/bash
# Environment template for the x86 export host (desktop TRT >= 10.5).
# Copy to your workspace, fill in the placeholders, and `source` it before
# running export/gate steps on x86.

export QD_WORKSPACE=${QD_WORKSPACE:?set QD_WORKSPACE, e.g. /home/<you>/qd-workspace}
export QD_UPSTREAM_REPO=$QD_WORKSPACE/qwen-drive           # upstream inference code (pip install -e .)
export QD_MODEL_DIR=$QD_WORKSPACE/Qwen-Drive-1.0-4B        # upstream weights
export QD_DATA_ROOT=$QD_WORKSPACE/pai_data
export QD_PAI_ROOT=$QD_DATA_ROOT                # PhysicalAI AV chunk 0 root (adapter input; ckpts/README.md)
export QD_ALPAMAYO_RECIPES=$QD_WORKSPACE/alpamayo-recipes  # PAI dataset interface used by eval/adapter
export QD_RUNTIME_DIR=$QD_WORKSPACE/runtime

# TensorRT-Edge-LLM checkout used by the exporter driver (v0.9.1)
export QD_EDGELLM_SRC=$QD_WORKSPACE/TensorRT-Edge-LLM

# CUDA toolkit for torch-extension builds (kernels/ parity harness on x86)
export QD_CUDA_HOME=${QD_CUDA_HOME:-/usr/local/cuda}

# Exporter python stack: torch + transformers (qwen3_5) + onnx; see
# docs/environment.md. On TRT >= 10.5 hosts the stock v0.9.1 tree builds as-is.
