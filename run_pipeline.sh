#!/usr/bin/env bash
# Pipeline driver for qwen-drive-edge-orin.
#
# Runs the certified pipeline in dependency order with fail-fast guards.
# Engine builds (vision trtexec / text Edge-LLM) are thin build commands
# documented in configs/*.env and export/patches/orin_text_engine/README.md;
# their guards below fail with a build pointer when the outputs are missing.
# Chain gate: production planner is the ONE-STEP TRT bf16 engine
# (run_planner_fastlane.sh: base all-torch 1-step arm vs engine arm with
# vision + planner engines; text stays eager because the text engine and the
# vision/planner engines must never share one process — multi-engine
# co-residency corrupts small tensors, the runner hard-rejects the
# combination). The archived 10-step eager-planner chain gate runs with
# QD_FAST_LANE=0. The vision engine is built + guarded and consumed by its
# own gate (gate_vision_edgellm.py).
# Stage-by-stage manual: docs/pipeline_guide.md
#
# Usage:
#   ./run_pipeline.sh <env.sh> [x86|orin|all]    # stage defaults to all
#
# Steps are idempotent: guards re-check on every run, completed export/gate
# outputs are reused when present.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${2:-all}"

if [[ $# -lt 1 || ! $STAGE =~ ^(x86|orin|all)$ ]]; then
    echo "usage: $0 <env.sh> [x86|orin|all]" >&2
    echo "  <env.sh>: filled copy of scripts/env_x86.sh or scripts/env_orin.sh" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "$1"

step() { printf '\n==== [step %s] %s ====\n' "$1" "$2"; }
die() { echo "FAIL: $*" >&2; exit 1; }

# --------------------------------------------------------------- preflight --
step 1 "preflight: workspace env"
: "${QD_WORKSPACE:?FAIL: source a filled copy of scripts/env_x86.sh or scripts/env_orin.sh first}"
for v in QD_MODEL_DIR QD_UPSTREAM_REPO QD_RUNTIME_DIR QD_DATA_ROOT QD_PAI_ROOT QD_ALPAMAYO_RECIPES; do
    [[ -n "${!v:-}" ]] || die "$v is not set (see scripts/env_*.sh)"
done
mkdir -p "$QD_RUNTIME_DIR"

step 2 "preflight: upstream artifacts (link-only; ckpts/README.md)"
[[ -f $QD_MODEL_DIR/config.json ]] \
    || die "upstream weights missing at \$QD_MODEL_DIR — run scripts/fetch_upstream.sh (ckpts/README.md)"
ls "$QD_MODEL_DIR"/*.safetensors >/dev/null 2>&1 \
    || die "no *.safetensors under \$QD_MODEL_DIR — run scripts/fetch_upstream.sh (ckpts/README.md)"
[[ -f $QD_UPSTREAM_REPO/src/qwen_drive/__init__.py ]] \
    || die "upstream inference code missing at \$QD_UPSTREAM_REPO — git clone https://github.com/QwenLM/qwen-drive"
ls "$QD_PAI_ROOT"/data/chunk-*/*.parquet >/dev/null 2>&1 \
    || ls "$QD_PAI_ROOT"/*.parquet >/dev/null 2>&1 \
    || die "PAI chunk 0 parquet files missing under \$QD_PAI_ROOT — gated download per ckpts/README.md"

step 3 "preflight: python stack (torch/tensorrt/transformers)"
"$QD_PY" - <<'EOF'
import sys
missing = []
for mod in ("torch", "tensorrt", "transformers"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    sys.exit("missing python modules: " + ", ".join(missing)
             + " (stack fingerprint: docs/environment.md)")
print("python stack OK:", end="")
import torch
print(" torch", torch.__version__, end="")
try:
    import tensorrt
    print(" tensorrt", tensorrt.__version__, end="")
except ImportError:
    pass
print()
EOF

# ---------------------------------------------------------------- x86 stage --
if [[ $STAGE == x86 || $STAGE == all ]]; then
    step 4 "x86: data adapter (PAI chunk 0 -> scenes jsonl + frames)"
    DATA_OUT="$QD_RUNTIME_DIR/data"
    SCENES="$DATA_OUT/qwen_scenes_chunk0.jsonl"
    if [[ -f $SCENES ]]; then
        echo "exists, reusing: $SCENES"
    else
        "$QD_PY" "$REPO_ROOT/eval/adapter/pai_to_qwen.py" \
            --out "$DATA_OUT" --num-scenes 94 --pack
    fi

    step 5 "x86: vision ONNX export + ORT parity gate"
    ONNX_OUT="$QD_RUNTIME_DIR/vision_export"
    if [[ -f $ONNX_OUT/vision_fp32.onnx && -f $ONNX_OUT/gate_vision_onnx_x86.json ]]; then
        echo "exists, reusing: $ONNX_OUT/vision_fp32.onnx"
    else
        mkdir -p "$ONNX_OUT"
        "$QD_PY" "$REPO_ROOT/export/vision/export_vision_onnx.py" \
            --output "$ONNX_OUT/vision_fp32.onnx" \
            --tensors "$ONNX_OUT/vision_test_tensors.npz"
        "$QD_PY" "$REPO_ROOT/export/vision/dump_edgellm_gate_inputs.py" \
            --tensors "$ONNX_OUT/vision_test_tensors.npz" --out "$ONNX_OUT/gate_inputs.npz"
        "$QD_PY" "$REPO_ROOT/export/vision/gate_vision_onnx_x86.py" \
            --onnx "$ONNX_OUT/vision_fp32.onnx" \
            --tensors "$ONNX_OUT/vision_test_tensors.npz" \
            --output "$ONNX_OUT/gate_vision_onnx_x86.json"
    fi

    step 6 "x86: text gate reference (validate golden + derive gate ref npz)"
    LOGS_TEXT="$QD_RUNTIME_DIR/edgellm/logs_text"
    GOLDEN="$LOGS_TEXT/mrope_golden_f2.npz"
    BUNDLE="$LOGS_TEXT/text_ref_bundle_f2.npz"
    if [[ -f $GOLDEN && -f $BUNDLE ]]; then
        mkdir -p "$LOGS_TEXT"
        "$QD_PY" "$REPO_ROOT/export/text/gen_mrope_ref_f2.py"
    else
        die "gate reference npz missing under $LOGS_TEXT -- hf download hous-lab/qwen-drive-edge-orin (ckpts/README.md)"
    fi
    step 5b "x86: planner one-step ONNX export (production planner lane)"
    if [[ ${QD_FAST_LANE:-1} == 1 ]]; then
        PLANNER_OUT="$QD_RUNTIME_DIR/planner_export"
        mkdir -p "$PLANNER_OUT"
        if [[ -f $PLANNER_OUT/planner_dyn.onnx && -f $PLANNER_OUT/export_pass.marker ]]; then
            echo "exists, reusing: $PLANNER_OUT/planner_dyn.onnx"
        else
            "$QD_PY" "$REPO_ROOT/export/planner/export_planner_onnx.py" \
                --model-dir "$QD_MODEL_DIR" --planner "$QD_MODEL_DIR/planner-rl" --dynamic \
                --out "$PLANNER_OUT/planner_dyn.onnx" 2>&1 | tee "$PLANNER_OUT/export.log"
            grep -q "^EXPORT PASS" "$PLANNER_OUT/export.log" \
                || die "planner ONNX export gate A/B not PASS (export.log)"
            touch "$PLANNER_OUT/export_pass.marker"
        fi
    fi

fi

# --------------------------------------------------------------- orin stage --
if [[ $STAGE == orin || $STAGE == all ]]; then
    step 7 "orin: Edge-LLM plugin build (GDN sm_87 kernel)"
    if [[ -f $QD_PLUGIN_SO ]]; then
        echo "exists, reusing: $QD_PLUGIN_SO"
    else
        bash "$REPO_ROOT/export/patches/orin_text_engine/build_plugin_orin.sh"
    fi

    step 8 "orin: GDN kernel parity gate"
    TORCH_CUDA_ARCH_LIST=8.7 "$QD_PY" "$REPO_ROOT/kernels/parity_tco.py"

    step 9 "orin: engine presence guards"
    VISION_ENGINE="${QD_VISION_ENGINE:-$QD_RUNTIME_DIR/vision.plan}"
    TEXT_ENGINE="${QD_TEXT_ENGINE:-$QD_RUNTIME_DIR/text.plan}"
    [[ -f $VISION_ENGINE ]] \
        || die "vision engine missing at $VISION_ENGINE — build with the trtexec flags in configs/vision_engine_build.env"
    [[ -f $TEXT_ENGINE ]] \
        || die "text engine missing at $TEXT_ENGINE — build per export/patches/orin_text_engine/README.md"

    step 10 "orin: archived 10-step chain gate (runs with QD_FAST_LANE=0)"
    if [[ ${QD_FAST_LANE:-1} == 0 ]]; then
    CHAIN_OUT="$QD_RUNTIME_DIR/gates/chain"
    if [[ -f $CHAIN_OUT/eval_v3_summary.json ]]; then
        echo "exists, reusing: $CHAIN_OUT/eval_v3_summary.json"
    else
        mkdir -p "$CHAIN_OUT"
        "$QD_PY" "$REPO_ROOT/runtime/chain/chain_runner.py" \
            --model-dir "$QD_MODEL_DIR" \
            --planner "$QD_MODEL_DIR/planner-rl" \
            --scenes "$QD_RUNTIME_DIR/data/qwen_scenes_chunk0.jsonl" --image-root "$QD_RUNTIME_DIR/data" \
            --num-steps 10 --kvlen 4096 --dtype bf16 \
            --output "$CHAIN_OUT/base.jsonl"
        "$QD_PY" "$REPO_ROOT/runtime/chain/chain_runner.py" \
            --model-dir "$QD_MODEL_DIR" \
            --planner "$QD_MODEL_DIR/planner-rl" \
            --scenes "$QD_RUNTIME_DIR/data/qwen_scenes_chunk0.jsonl" --image-root "$QD_RUNTIME_DIR/data" \
            --num-steps 10 --kvlen 4096 --dtype bf16 \
            --text-engine "$TEXT_ENGINE" \
            --output "$CHAIN_OUT/eng.jsonl"
        "$QD_PY" "$REPO_ROOT/runtime/chain/eval_chain_v3.py" "$CHAIN_OUT/base.jsonl" "$CHAIN_OUT/eng.jsonl" "$CHAIN_OUT/eval_v3_summary.json" "eng"
    fi
    fi

    step 9b "orin: planner engine build + gate (production planner lane)"
    if [[ ${QD_FAST_LANE:-1} == 1 ]]; then
        # shellcheck disable=SC1091
        source "$REPO_ROOT/configs/planner_engine_build.env"
        PLANNER_OUT="$QD_RUNTIME_DIR/planner_export"
        PLAN="$PLANNER_OUT/planner_one_step_dyn_bf16.plan"
        if [[ ! -f $PLANNER_OUT/planner_gate_bundle_s1141.npz ]]; then
            "$QD_PY" "$REPO_ROOT/export/planner/dump_gate_bundle.py" \
                --model-dir "$QD_MODEL_DIR" --planner "$QD_MODEL_DIR/planner-rl" --scene-len 1141 \
                --out "$PLANNER_OUT/planner_gate_bundle_s1141.npz"
        fi
        if [[ ! -f $PLAN ]]; then
            QD_PLANNER_ONNX="$PLANNER_OUT/planner_dyn.onnx" \
                bash "$REPO_ROOT/export/planner/build_planner_engine.sh"
        fi
        mkdir -p "$QD_RUNTIME_DIR/gates/planner"
        "$QD_PY" "$REPO_ROOT/export/planner/gate_planner_engine.py" \
            --plan "$PLAN" --bundle "$PLANNER_OUT/planner_gate_bundle_s1141.npz" \
            --scene-len 1141 | tee "$QD_RUNTIME_DIR/gates/planner/gate_planner_engine.log"
        grep -q "GATE PASS" "$QD_RUNTIME_DIR/gates/planner/gate_planner_engine.log" \
            || die "planner engine gate not PASS (gate_planner_engine.log)"
    fi

    step 10b "orin: planner chain gate (production)"
    if [[ ${QD_FAST_LANE:-1} == 1 ]]; then
        QD_PLANNER_ENGINE="$PLAN" bash "$REPO_ROOT/runtime/chain/run_planner_fastlane.sh"
    fi

fi
step 99 "done"
echo "All requested stages completed. Gate criteria: configs/manifest_f2_05.env"
