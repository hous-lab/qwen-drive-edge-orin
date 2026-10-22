#!/usr/bin/env bash
# Download helper for the upstream artifacts listed in ckpts/README.md.
# Upstream-hosted files are LINK-ONLY by policy: this repo never ships or
# rehosts them. This script only fetches to the workspace layout.
set -euo pipefail

WS=${QD_WORKSPACE:?source scripts/env_*.sh first}
MODELS_DIR=$QD_MODEL_DIR
DATA_ROOT=$QD_DATA_ROOT

echo "== 1. upstream inference code =="
if [[ ! -d $QD_UPSTREAM_REPO/src/qwen_drive ]]; then
    git clone https://github.com/QwenLM/qwen-drive "$QD_UPSTREAM_REPO"
fi
echo "(pip install -e $QD_UPSTREAM_REPO)  # run manually per docs/environment.md"

echo "== 2. model weights (~32.5 GB) =="
REPO_ID=Qwen/Qwen-Drive-1.0-4B
if [[ ! -d $MODELS_DIR ]]; then
    pip install -U "huggingface_hub[cli]"
    hf download $REPO_ID --local-dir "$MODELS_DIR"
fi

echo "== 3. PAI chunk 0 (gated dataset; license acceptance required) =="
echo "Download after accepting the NVIDIA AV Dataset License at"
echo "  https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles"
echo "Place files under $DATA_ROOT per ckpts/README.md."

echo "== done. Verify sha256 against ckpts/README.md =="
