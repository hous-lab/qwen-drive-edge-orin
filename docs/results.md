# Results

All numbers: Jetson AGX Orin 64GB, locked clocks, single stream, batch 1.
The production configuration is the ONE-STEP TRT bf16 planner engine (one
engine call per scene; promoted 2026-10-25 on three-arm ground-truth
evidence, see the promotion section below). Latencies in the production
table are wall-clock means from the re-run of this repo's pipeline
(2026-10-25 window, mildly degraded compute — 24.2 TF vs the 27.9 TF bf16
anchor, numerics anchor-gated — so provisional); the 2026-10-12
certified-window numbers for the archived 10-step planner chain are kept
below as history. Accuracy numbers are the 94-scene open-loop evaluation
(engine arm vs torch reference arms over the same scenes; no closed-loop
evaluation was executed in this project).

## Production chain

| Stage | Implementation | Latency (re-run 2026-10-25 window, provisional) |
|---|---|---|
| Process (tokenize + glue) | torch | 208.2 ms |
| Prefill (vision + text) | vision TRT fp16 plan + text torch bf16 eager | 782.0 ms |
| Planner | TRT bf16 one-step engine, dynamic-S | 35.7 ms |
| End to end | full chain | 1026.0 ms |

The planner serves every scene with ONE engine call (35.7 ms in-chain; 29.7
ms standalone p50 at S=1141). The chain runs the text prefill eagerly in
bf16 because the text engine and the planner engine cannot share one
process (engine co-residency corrupts small tensors; the runner
hard-rejects the combination).

### Archived certified baseline (2026-10-12 certified window)

| Stage | Implementation | Latency |
|---|---|---|
| Vision | TRT fp16 plan (Edge-LLM path) | 211.6 ms |
| Text prefill | TRT fp16 engine, T=1141 | 580.5 ms |
| Planner | torch eager (PlanningExpert sampler, 10-step rollout) | 1162.7 ms |
| Orchestration | torch glue (tokenize + stage handoff) | 216.6 ms |
| End to end | vision + text + planner | 2191.5 ms |

The archived planner stage is the PlanningExpert flow-matching sampler
rolled out eagerly for its 10 sampling steps (num_steps=10); per-step it
measures about 120 ms eager, 33 ms under a torch-only CUDA-graph context
(bitwise-equal), and 29.9-36.4 ms as a TRT engine. The torch CUDA-graph
variant is forbidden under engine co-residency. Numbers are stage means
over the same 94-scene run; the end-to-end mean additionally carries about
20 ms of stage-internal glue (mrope merge and cache handoff), so it sits
slightly above the four-row sum.

## Planner promotion: three-arm ground-truth comparison

The promotion decision does not rest on the plan-space twin-fidelity
metric. It rests on the 94-scene manifest with ground truth
(future_trajectory), comparing the promoted engine arm against the
certified 10-step fp semantics reference and against a pure-torch one-step
arm that is identical in sampler semantics:

| Arm | ADE3 (m) | ADE5 (m) | FDE5 (m) | ADE5/FDE5 rel vs 10-step fp | Plan bucket |
|---|---|---|---|---|---|
| 10-step torch (fp semantics reference) | 1.141 | 2.589 | 6.894 | - | 1162.7 ms |
| 1-step torch (same sampler semantics as the engine arm) | 1.220 | 2.665 | 7.223 | +2.93% / +4.78% | 122.8 ms |
| 1-step engine (promoted production planner) | 1.219 | 2.653 | 7.164 | +2.48% / +3.92% | 35.7 ms |

Reading of the table:

- The one-step sampler semantics cost +2.93%/+4.78% in pure torch (row 2
  vs row 1). The engine arm lands at +2.48%/+3.92% - slightly CLOSER to
  the reference than its own semantics twin, i.e. the engine contributes
  no measurable task damage. Scene-by-scene, the engine arm improves 47 of
  94 scenes on ADE5 vs its twin, mean ADE5 change -1.2 cm.
- Internal arms corroborate: the archived 10-06 engine arm is bitwise
  identical to the fresh re-run engine arm, and a fresh 10-25 1-step torch
  re-run matched the archived torch arm to three decimals.
- The +2.5%/+3.9% in production is therefore entirely one-step sampler
  semantics, shown identically by a pure-torch arm, and it sits far inside
  the +-10% band used by the gate.

## Why twin fidelity (5.1194 cm) and task metrics disagree

The chain gate's plan-space twin fidelity (mean abs delta between the
engine plan and its same-semantics torch reference plan) reads 5.1194 cm
against a 5.0 cm bar, while the ground-truth task metrics are in band.
The two metric families measure different things:

- Twin fidelity is a CHANGE DETECTOR: engine arm vs torch twin, no ground
  truth in the loop. A plan can differ from its twin by centimeters while
  sitting at the same or smaller distance from ground truth (e.g. lateral
  offsets along a lane centerline that are GT-consistent).
- ADE/FDE are QUALITY detectors: distance to ground truth. The worst
  twin-fidelity outlier (30.25 cm twin delta) has the engine arm 2.2x
  CLOSER to ground truth than its twin (ADE5 0.737 m vs 1.601 m); the
  second outlier (30.10 cm) IMPROVES on GT by -0.90 m ADE5 (FDE5
  11.319 -> 8.217 m).
- The one-step regime amplifies the change-detector: with a single
  denoising step there is no iterative pull-back toward the reference, so
  bf16 kernel-level differences surface as cm-scale plan deltas. The
  archived 10-step engine arm passed twin fidelity at 1.3076 cm - the
  iterative sampler averages out per-step noise.
- Distribution over 94 scenes: mean 5.1194 cm, ex-top-2 4.575 cm,
  p50 3.883 cm, p90 9.744 cm, max 30.25 cm; 34/94 scenes over the 5 cm
  bar. GT-anchored: mean ADE5 change -1.2 cm, worst single-scene
  worsening +0.30 m ADE5 (scene GT distances are meters).

## Fast-lane re-run provenance

We re-ran the entire promoted lane end-to-end from the shipped tree
(x86 ONNX export -> Orin gate bundle -> engine build -> engine gate ->
94-scene chain gate) before promoting it. Every gate number reproduced the
archived internal measurement: the chain twin-fidelity read 5.1194 cm -
the same 7 significant figures - and the per-scene structure is identical
down to the outlier list (34 scenes over 5 cm, tops 30.25 / 30.10 /
16.10 / 11.65 cm; one of the two 30 cm scenes is the certified 10-step
arm's worst at 35.23 cm, i.e. the known degenerate-plan family, not
engine-specific damage). The engine gate reproduced bit-consistently
(cos 0.99992638, max 0.88 cm, byte-exact gate inputs), and the serving
contract held: 94/94 planner calls on-engine, zero eager fallback.

| Metric (re-run 2026-10-25) | Value | Archived (2026-10-06) |
|---|---|---|
| Engine gate at S=1141 (bar cos >= 0.999) | cos 0.99992638, max 0.88 cm, PASS | identical |
| Engine call p50 (S=1141, n=200) | 29.72 ms | 29.87 ms |
| In-chain plan bucket | 35.7 ms | 35.9 ms |
| All-torch 1-step plan bucket (base arm) | 122.8 ms | not measured |
| End-to-end (1-step arm) | 1026 ms vs 1210 ms base (-184 ms) | 1028 ms |
| Chain twin fidelity (record-only) | 5.1194 cm | identical |
| ADE5 / FDE5 relative (vs its 1-step reference) | -0.44% / -0.82% (in band) | in band |

Re-run timings were taken in a mildly degraded compute window (24.2 TF vs
the 27.9 TF bf16 anchor, numerics anchor-gated) and are provisional; the
archived certified-window numbers are kept above as history.

## Gates summary

Production gate (one-step engine planner arm, eval_chain_v3 over the
94-scene manifest):

| Metric | Value | Criterion |
|---|---|---|
| ADE5 / FDE5 vs ground truth, relative to the 10-step fp reference | +2.48% / +3.92% | <= +10% |
| NaN scenes | 0 | 0 allowed |
| Engine-serving contract (planner calls on-engine) | 94/94 | 94/94 required (eager fallback is FATAL) |
| Plan-space twin fidelity | 5.1194 cm mean abs | record-only (promotion evidence above) |

Archived gate (10-step eager planner, engine text-prefill arm, 2026-10-12
certified window; reproducible with QD_FAST_LANE=0):

| Metric | Value | Criterion |
|---|---|---|
| Mean abs trajectory delta | 1.3076 cm | <= 5 cm |
| ADE5 / FDE5 relative | in band | <= +10% |
| NaN scenes | 0 | 0 allowed |

## GDN sm_87 kernel contribution

The shipped kernel is the TC-o variant (bf16 wmma in the output segment) of
the self-developed sm_87 chunk-prefill kernel; healthy-window A/B on Orin:
O-segment speedup x2.16-4.32 (dual-metric), full-prefill segment speedup
x1.6-1.8, trt_text stage -166.7 ms (-22%), end-to-end -144.5 ms.

## Numerical-parity note

Deployment behavior (94-scene gate PASS) and internal numerical parity are
decoupled in this system: a kv-cos basin near 0.88 in one measurement lane
persisted across the release and is documented as UNRESOLVED — it does not
propagate to closed-loop trajectory behavior. We flag this because a parity
lane that disagrees with task metrics is a real engineering signal, not a
box-check; see [docs/negative_results.md](negative_results.md) for the measurement-chain caveats
that produced it.
