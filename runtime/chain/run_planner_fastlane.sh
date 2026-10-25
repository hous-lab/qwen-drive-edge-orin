#!/usr/bin/env bash
# run_planner_fastlane.sh — optional one-step planner engine lane over the
# certified chain-gate design: fresh all-torch 1-step goldens (base arm), then
# the engine arm (vision + planner engines; text stays eager because
# chain_runner hard-rejects planner+text engines in one process — engine
# co-residency), then eval_chain_v3. The trajectory gate is RECORD-ONLY for
# this lane (documented marginal miss in docs/results.md); the engine-serving
# contract is FATAL.
#
# Consumes the scripts/env_orin.sh contract plus:
#   QD_VISION_ENGINE   vision engine (configs/vision_engine_build.env path)
#   QD_PLANNER_ENGINE  planner one-step engine (build_planner_engine.sh output)
# Gate semantics: mean absolute trajectory delta vs the SAME-SEMANTICS
# all-torch 1-step reference over the 94-scene manifest; bar 5.0 cm.
set -euo pipefail

: "${QD_MODEL_DIR:?set QD_MODEL_DIR}"
: "${QD_RUNTIME_DIR:?set QD_RUNTIME_DIR}"
: "${QD_PY:?set QD_PY}"
: "${QD_VISION_ENGINE:?set QD_VISION_ENGINE to the vision engine}"
: "${QD_PLANNER_ENGINE:?set QD_PLANNER_ENGINE to the planner one-step engine}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CH="$REPO/runtime/chain"
OUT="$QD_RUNTIME_DIR/gates/planner_chain"
mkdir -p "$OUT"

SCENES="${QD_SCENES:-$QD_RUNTIME_DIR/data/qwen_scenes_chunk0.jsonl}"
[[ -f $SCENES ]] || { echo "FATAL: scenes jsonl missing at $SCENES (run the pipeline data step)" >&2; exit 1; }
[[ -f $QD_VISION_ENGINE ]] || { echo "FATAL: vision engine missing at $QD_VISION_ENGINE" >&2; exit 1; }
[[ -f $QD_PLANNER_ENGINE ]] || { echo "FATAL: planner engine missing at $QD_PLANNER_ENGINE" >&2; exit 1; }

echo "== fast-lane base arm: all-torch 1-step goldens"
"$QD_PY" "$CH/chain_runner.py" \
    --model-dir "$QD_MODEL_DIR" \
    --planner "$QD_MODEL_DIR/planner-rl" \
    --scenes "$SCENES" --image-root "$QD_DATA_ROOT" \
    --num-steps 1 --kvlen 4096 --dtype bf16 \
    --output "$OUT/base_1step.jsonl" 2>&1 | tee "$OUT/stage_base.log"
grep -q "AVG_MS" "$OUT/stage_base.log" || { echo "FATAL: base arm produced no AVG_MS" >&2; exit 1; }

echo "== fast-lane eng arm: vision + planner engines (text eager)"
"$QD_PY" "$CH/chain_runner.py" \
    --model-dir "$QD_MODEL_DIR" \
    --planner "$QD_MODEL_DIR/planner-rl" \
    --scenes "$SCENES" --image-root "$QD_DATA_ROOT" \
    --num-steps 1 --kvlen 4096 --dtype bf16 \
    --vision-engine "$QD_VISION_ENGINE" \
    --planner-engine "$QD_PLANNER_ENGINE" \
    --output "$OUT/eng_pe1.jsonl" 2>&1 | tee "$OUT/stage_eng.log"
grep -q "AVG_MS" "$OUT/stage_eng.log" || { echo "FATAL: eng arm crashed" >&2; exit 1; }

# Contract: the planner engine must serve EVERY call (disabled or eager
# fallback would silently degrade to torch semantics; both fatal here).
PE=$(grep -a "^PLANNER_ENG" "$OUT/stage_eng.log" | tail -1)
echo "$PE"
echo "$PE" | grep -q "disabled=True" && { echo "CONTRACT FAIL: planner engine disabled" >&2; exit 1; }
echo "$PE" | grep -qE "eager_fallback=[1-9]" && { echo "CONTRACT FAIL: eager fallback" >&2; exit 1; }

echo "== gate (record-only for the optional lane)"
set +e
"$QD_PY" "$CH/eval_chain_v3.py" "$OUT/base_1step.jsonl" "$OUT/eng_pe1.jsonl" \
    "$OUT/eval_v3_summary.json" "pe1"
GATE_RC=$?
set -e
if [[ $GATE_RC == 0 ]]; then
    echo "FAST-LANE GATE: PASS"
else
    echo "FAST-LANE GATE: FAIL (record-only for the optional lane; docs/results.md)"
fi
echo "outputs under $OUT"
