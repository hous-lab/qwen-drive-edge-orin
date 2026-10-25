# Results

All numbers: Jetson AGX Orin 64GB (sm_87), MAXN power mode, clocks locked (`jetson_clocks`), single stream, batch 1. Software stack pinned in [docs/environment.md](environment.md).

## Measurement metadata

| Field | Value |
|---|---|
| Device | Jetson AGX Orin 64GB (sm_87) |
| Power / clocks | MAXN, clocks locked with `jetson_clocks` |
| Software stack | torch 2.8.0, TensorRT 10.4, TensorRT-Edge-LLM v0.9.1 + shipped patch bundle; exact versions in [docs/environment.md](environment.md) |
| Harness | [run_pipeline.sh](../run_pipeline.sh); every number in this page regenerates from this repo's pipeline |
| Latency semantics | stage and end-to-end latencies are means over one 94-scene open-loop pass (one inference per scene, pre/post-processing included); the planner-engine standalone number is p50 over 200 timed calls after 20 warmup calls with inputs resident ([gate_planner_engine.py](../export/planner/gate_planner_engine.py)) |
| Accuracy semantics | open-loop, ground-truth-anchored: ADE3 = mean distance over the first 30 points, ADE5 = mean over all 50 points, FDE5 = last point of the 10 Hz 50-point future trajectory ([eval_chain_v3.py](../runtime/chain/eval_chain_v3.py)); one inference per scene, no closed-loop rollouts |
| Reproducibility | independent full-lane runs of this repo's pipeline reproduce the per-scene records to 7 significant figures; engine-gate inputs are bitwise-identical across runs |

## Chain dataflow

Camera frames (three views × four frames) are encoded by the vision TRT fp16 engine into vision features; the text prefill embeds the scene record (history frames, ego status, driving commands) plus the vision features with mrope and produces KV for the planner's context stage; the planner engine consumes the KV and emits the 50-point 10 Hz trajectory in one engine call. The baseline lane replaces the eager text prefill with a TensorRT-Edge-LLM text engine whose GDN layers run the shipped sm_87 chunk-prefill kernel.

## Production chain

| Stage | Implementation | Latency |
|---|---|---|
| Process (tokenize + glue) | torch | 208.2 ms |
| Prefill (vision + text) | vision TRT fp16 engine + text torch bf16 eager | 782.0 ms |
| Runner | TRT bf16 one-step engine, dynamic-S | 35.7 ms |
| End to end | full chain | 1026.0 ms |

The planner serves every scene with ONE engine call (35.7 ms in-chain; 29.7 ms standalone p50 at S=1141). The chain runs the text prefill eagerly in bf16 because the text engine and the planner engine cannot share one process — multi-engine co-residency corrupts small tensors and the runner hard-rejects the combination (documented in [negative_results.md](negative_results.md)).

## Baseline and arm design

The baseline on the same device is the same chain with the planner run eagerly through its native 10-step sampler rollout and the text prefill served by the TRT text engine: vision TRT 211.6 ms + text engine 580.5 ms + planner 1162.7 ms + orchestration 216.6 ms = 2191.5 ms end to end. It passes all gates at 1.3076 cm and remains reproducible with `QD_FAST_LANE=0` — this is the configuration to compare against when evaluating future planner variants.

Accuracy is measured as a three-arm comparison on the 94-scene manifest with ground truth:

| Arm | Definition |
|---|---|
| 10-step torch | pure-torch bf16 eager implementation of the model's native 10-step sampler; the fp semantics reference |
| 1-step torch | pure-torch implementation of the one-step sampler; the engine arm's semantics twin |
| 1-step engine | the production TRT engine arm; engine implementation of the one-step sampler |

## Task-level accuracy

| Arm | ADE3 (m) | ADE5 (m) | FDE5 (m) | vs 10-step reference | Planner latency |
|---|---|---|---|---|---|
| 10-step torch | 1.141 | 2.589 | 6.894 | — | 1162.7 ms |
| 1-step torch | 1.220 | 2.665 | 7.223 | +2.93% / +4.78% | 122.8 ms |
| 1-step engine (production) | 1.219 | 2.653 | 7.164 | **+2.48% / +3.92%** | 35.7 ms |

The engine adds no measurable task damage: against its own semantics twin it is marginally closer to ground truth (ADE5 2.653 vs 2.665 m, FDE5 7.164 vs 7.223 m). Scene-level, the engine arm improves 47 of 94 scenes on ADE5 vs its twin, with a mean ADE5 change of −1.2 cm and a worst single-scene worsening of +0.30 m (GT distances on this manifest are meters).

The +2.5%/+3.9% vs the 10-step reference is one-step sampler semantics, not engine damage: the pure-torch one-step arm shows it identically (+2.93%/+4.78%), and the engine arm lands marginally closer to the reference than its own twin. It sits far inside the ±10% band used by the gate. An independent 1-step torch run matched the torch arm to three decimals, and the engine arm reproduces bitwise across independent full-lane runs.

## Twin fidelity: change detector vs quality detector

The chain gate also records plan-space twin fidelity — the mean absolute delta between the engine plan and its same-semantics torch twin, with no ground truth in the loop. Production reads 5.1194 cm mean abs (p50 3.883, p90 9.744, max 30.25 cm; 34/94 scenes over the 5.0 cm bar).

Two facts stop this from contradicting the in-band task metrics:

- The worst twin-fidelity outlier (30.25 cm) has the engine arm 2.2× closer to ground truth than its twin (ADE5 0.737 m vs 1.601 m); the second outlier (30.10 cm) improves on GT by −0.90 m ADE5 (FDE5 11.319 → 8.217 m). One of the two is also the 10-step reference arm's worst scene at 35.23 cm — the known degenerate-plan family, not engine-specific damage.
- The one-step regime amplifies the change-detector: with a single denoising step there is no iterative pull-back toward the reference, so bf16 kernel-level differences surface as centimeter-scale plan deltas. The baseline's 10-step engine arm reads 1.3076 cm on the same metric because the iterative sampler averages out per-step noise.

Accordingly, twin fidelity is a change detector, not a quality metric: it is recorded for the production gate but not gated on it; go/no-go rests on the ground-truth-anchored metrics plus the engine-serving contract. Whether any per-scene twin delta is GT-harmful is decided by the GT metrics, and those are in band.

## Gates

Production gate (one-step engine planner arm):

| Metric | Value | Criterion |
|---|---|---|
| ADE5 / FDE5 vs ground truth, relative to the 10-step reference | +2.48% / +3.92% | <= +10% |
| NaN scenes | 0 | 0 allowed |
| Engine-serving contract (planner calls on-engine) | 94/94 | 94/94 required (eager fallback is FATAL) |
| Plan-space twin fidelity | 5.1194 cm mean abs | record-only |

Baseline gate (10-step eager planner, text engine lane; `QD_FAST_LANE=0`):

| Metric | Value | Criterion |
|---|---|---|
| Mean abs trajectory delta (engine arm vs torch reference) | 1.3076 cm | <= 5 cm |
| ADE5 / FDE5 relative | in band | <= +10% |
| NaN scenes | 0 | 0 allowed |

## GDN kernel contribution

The shipped kernel is a chunk-parallel prefill kernel for the GDN layers on sm_87, with bf16 wmma in the output segment, patched into TensorRT-Edge-LLM. Measured in the baseline text-engine lane: O-segment speedup ×2.16-4.32 (dual-metric), full-prefill segment ×1.6-1.8, trt_text stage −166.7 ms (−22%), 10-step chain end-to-end −144.5 ms. The production chain runs text eagerly, so the kernel is exercised by the baseline text-engine lane and ships for that lane; consumers who run the text-engine lane inherit it.

## Numerical-parity note (UNRESOLVED)

Deployment behavior and one internal numerical-parity lane are decoupled in this system.

- FACT: on the same inputs where the planner-engine gate reads cos 0.99993 and the chain task metrics are in band, the parity lane reads kv-cos ≈ 0.88.
- INTERPRETATION: checkpoint forensics found all four checkpoints along the lane clean, and the observed 0.88-0.98 band matches the interval spanned by pre-vs-post-rope construction of the KV — so gate comparison semantics (what is compared to what) is the leading suspect, ahead of proven numerical corruption, which was never demonstrated.
- HYPOTHESIS (unverified): the lane applies the rope table to the KV under a different convention than the reference; confirming or refuting requires re-deriving the lane's reference construction.
- The lane reads low on every run and never propagated to task metrics across the release; we flag it because a parity lane that disagrees with task metrics is a real engineering signal, not a box-check. The measurement-chain caveats that produced it are in [docs/negative_results.md](negative_results.md).

## Limitations

- Open-loop only: no closed-loop evaluation was executed in this project; the environment pointer is in [ckpts/README.md](../ckpts/README.md).
- Bound to sm_87 and the pinned stack ([docs/environment.md](environment.md)); the kernel is sm_87-specific (wmma tile shapes, shared-memory budget).
- The 94-scene manifest is one PAI chunk 0 slice; this is a deployment study, not a driving-benchmark claim.
- Latencies were measured on a unit whose compute window runs mildly degraded (24.2 TF vs the 27.9 TF bf16 anchor on this GPU model); treat absolute values as unit-specific — the A/B deltas and gate verdicts are anchor-gated and cross-run stable to 7 significant figures.
- INT8 routes are documented as closed with root causes in [docs/negative_results.md](negative_results.md); the shipped chain is fp16/bf16.
- The parity lane above is UNRESOLVED and does not propagate to task metrics.
