#!/usr/bin/env bash
# build_planner_engine.sh — build the one-step planner TRT engine (bf16, dynamic
# scene length) from the exported ONNX with trtexec.
#
# The profile spans the certified manifests: r640 scenes sit at S in
# [1745..1750] and f2 scenes at S in [1139..1142] (scene-KV lengths). Consumers
# with a single manifest can tighten the profile with QD_PLANNER_S_MIN/MAX.
#
# Usage (Orin, after the x86 ONNX export):
#   QD_PLANNER_ONNX=<...>.onnx bash export/planner/build_planner_engine.sh
set -euo pipefail

: "${QD_RUNTIME_DIR:?set QD_RUNTIME_DIR (scripts/env_orin.sh)}"
: "${QD_PLANNER_ONNX:?set QD_PLANNER_ONNX to the planner ONNX from
export/planner/export_planner_onnx.py --dynamic (run on the x86 host)}"

TRTEXEC="${QD_TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
S_MIN="${QD_PLANNER_S_MIN:-1139}"
S_OPT="${QD_PLANNER_S_OPT:-1748}"
S_MAX="${QD_PLANNER_S_MAX:-1750}"

OUT_DIR="$QD_RUNTIME_DIR/planner_export"
mkdir -p "$OUT_DIR"
PLAN="$OUT_DIR/planner_one_step_dyn_bf16.plan"
LOG="$OUT_DIR/build_planner_engine.log"

[[ -f $QD_PLANNER_ONNX ]] || { echo "FATAL: $QD_PLANNER_ONNX missing" >&2; exit 1; }
[[ -x $TRTEXEC || -e $TRTEXEC ]] || { echo "FATAL: trtexec not at $TRTEXEC (set QD_TRTEXEC)" >&2; exit 1; }

# Model constants (Qwen-Drive-1.0-4B planning expert): 8 KV groups
# (num_hidden_layers/layers_per_kv) x 4 KV heads x 256 head_dim.
KV_GROUPS=8; KV_HEADS=4; HEAD_DIM=256

echo "building $PLAN (profile S[$S_MIN..$S_OPT..$S_MAX]); this takes ~9 min on Orin"
"$TRTEXEC" \
  --onnx="$QD_PLANNER_ONNX" \
  --saveEngine="$PLAN" \
  --minShapes=scene_k:${KV_GROUPS}x${S_MIN}x${KV_HEADS}x${HEAD_DIM},scene_v:${KV_GROUPS}x${S_MIN}x${KV_HEADS}x${HEAD_DIM} \
  --optShapes=scene_k:${KV_GROUPS}x${S_OPT}x${KV_HEADS}x${HEAD_DIM},scene_v:${KV_GROUPS}x${S_OPT}x${KV_HEADS}x${HEAD_DIM} \
  --maxShapes=scene_k:${KV_GROUPS}x${S_MAX}x${KV_HEADS}x${HEAD_DIM},scene_v:${KV_GROUPS}x${S_MAX}x${KV_HEADS}x${HEAD_DIM} \
  --bf16 --memPoolSize=workspace:2048 \
  --builderOptimizationLevel=3 --noBuilderCache \
  > "$LOG" 2>&1

echo "== plan:"; ls -la "$PLAN"
grep -E "Engine built in|Latency|GPU Compute Time" "$LOG" | tail -6
