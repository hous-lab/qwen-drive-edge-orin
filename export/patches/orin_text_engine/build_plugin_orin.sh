#!/usr/bin/env bash
# Build the Edge-LLM plugin + llm_build targets on Jetson AGX Orin with
# TensorRT-Edge-LLM v0.9.1 and the GDN sm_87 kernel patch applied. Run this ON THE ORIN.
#
# Inputs (env):
#   QD_EDGELLM_SRC   fresh TensorRT-Edge-LLM v0.9.1 checkout ON THE ORIN
#   QD_RUNTIME_DIR   runtime work dir (engines, logs) — created if missing
# Optional:
#   QD_BUILD_DIR     build tree name (default build_orin_text)
#
# Compat note: a stock v0.9.1 checkout targets TRT >= 10.5 (desktop). JetPack
# 6.x ships TRT 10.3 (libnvinfer) / 10.4 (python wheel), so the checkout needs
# the TRT-10.4 compat deltas before it configures: trtUtils.cpp
# (IStreamReaderV2 -> stream-API), cuda126_cutedsl_compat.h
# (force-included by kernels/plugin targets), and the Orin cmake/ fragments.
# These are carried by the production lineage trees; see
# docs/environment.md for the extraction procedure and md5 anchors.
set -euo pipefail

SRC=${QD_EDGELLM_SRC:?set QD_EDGELLM_SRC to the v0.9.1 checkout on the Orin}
RUNTIME=${QD_RUNTIME_DIR:-$(dirname "$SRC")/qd-runtime}
BUILD=${QD_BUILD_DIR:-build_orin_text}
HERE=$(cd "$(dirname "$0")" && pwd)
LOG_DIR=$RUNTIME/edgellm/logs_text
mkdir -p "$LOG_DIR"

echo "== 1. apply the patch bundle (idempotent) =="
rsync -a "$HERE/files/" "$SRC/"
cp -p "$HERE/0001-gdn-sm87-chunk-prefill-dispatch.patch" "$SRC/"
cd "$SRC"
if grep -q runGdnChunkPrefillSm87 cpp/kernels/gdnKernels/cuteDslGDNRunner.cpp; then
    echo "patch 0001 already applied"
else
    patch -p1 < 0001-gdn-sm87-chunk-prefill-dispatch.patch
fi

echo "== 2. stage the GDN kernel (canonical entry; shipped source md5 a065f1bf) =="
cp -p "$HERE/../../kernels/chunk_delta_cuda.cu" "$HERE/../../kernels/chunk_delta_cuda.h" \
    "$SRC/cpp/kernels/gdnKernels/"

echo "== 3. cmake configure + build =="
cd "$SRC"
mkdir -p "$BUILD" && cd "$BUILD"
PATH=/usr/local/cuda-12.6/bin:$PATH cmake .. \
    -DCMAKE_BUILD_TYPE=Release -DCUDA_CTK_VERSION=12.6 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
    -DENABLE_CUTE_DSL=ALL -DCMAKE_CUDA_ARCHITECTURES=87
PATH=/usr/local/cuda-12.6/bin:$PATH make -j"$(nproc)" llm_build NvInfer_edgellm_plugin

echo "== done =="
echo "plugin: $SRC/$BUILD/libNvInfer_edgellm_plugin.so"
echo "export QD_PLUGIN_SO=$SRC/$BUILD/libNvInfer_edgellm_plugin.so"
