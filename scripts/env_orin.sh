#!/bin/bash
# Environment template for Jetson AGX Orin (JetPack 6.x, L4T R36.5.0).
# Copy to your workspace, fill in the placeholders, and `source` it before
# running any pipeline step on the Orin.
# Full environment fingerprint: docs/environment.md

# --- workspace layout (consumer contract: side-by-side dirs) -------------
export QD_WORKSPACE=${QD_WORKSPACE:?set QD_WORKSPACE, e.g. /home/<you>/qd-workspace}
export QD_UPSTREAM_REPO=$QD_WORKSPACE/qwen-drive           # upstream inference code (pip install -e .)
export QD_MODEL_DIR=$QD_WORKSPACE/Qwen-Drive-1.0-4B        # upstream weights (see ckpts/README.md)
export QD_DATA_ROOT=$QD_WORKSPACE/pai_data                 # PAI chunk0 + regenerated scenes jsonl
export QD_PAI_ROOT=$QD_DATA_ROOT                           # PhysicalAI AV chunk 0 root (adapter input; ckpts/README.md)
export QD_ALPAMAYO_RECIPES=$QD_WORKSPACE/alpamayo-recipes  # PAI dataset interface used by eval/adapter
export QD_RUNTIME_DIR=$QD_WORKSPACE/runtime                # engines, gates, outputs (created as needed)

# --- Edge-LLM build tree (created by export/patches/.../build_plugin_orin.sh) ---
export QD_EDGELLM_SRC=$QD_WORKSPACE/TensorRT-Edge-LLM      # fresh v0.9.1 checkout + patch bundle
export QD_PLUGIN_SO=$QD_EDGELLM_SRC/build_orin_text/libNvInfer_edgellm_plugin.so

# --- python ---------------------------------------------------------------
# torch 2.8.0 + tensorrt 10.4 wheel + transformers (qwen3_5) per docs/environment.md
export QD_PY=${QD_PY:-python3}

# --- optional tuning ------------------------------------------------------
# QD_GDN_OFFICIAL=1 restores NVIDIA's official CuTe GDN kernel at runtime
# (all-zero output on sm_87 — kept as the documented fallback toggle).
# QD_CUDA_HOME: CUDA toolkit prefix for the parity-harness torch-extension build.
