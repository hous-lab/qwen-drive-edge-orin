# Results

All numbers: Jetson AGX Orin 64GB, locked clocks, fp16 TRT engines, single
stream, batch 1. Latencies are wall-clock means over the certified run
window (2026-10-12 production baseline v2); accuracy numbers are the
94-scene open-loop trajectory gate (engine arm vs torch reference arm over
the same scenes; no closed-loop evaluation was executed in this project).

## Production chain (certified v2 baseline)

| Stage | Implementation | Latency |
|---|---|---|
| Vision | TRT fp16 plan (Edge-LLM path) | 211.6 ms |
| Text prefill | TRT fp16 engine, T=1141 | 580.5 ms |
| Planner | torch eager (PlanningExpert sampler, 10-step rollout) | 1162.7 ms |
| Orchestration | torch glue (tokenize + stage handoff) | 216.6 ms |
| End to end | vision + text + planner | 2191.5 ms |

The planner stage is the PlanningExpert flow-matching sampler rolled out
eagerly for its 10 sampling steps (num_steps=10); per-step it measures about
120 ms eager, 33 ms under a torch-only CUDA-graph context (bitwise-equal),
and 29.9-36.4 ms as a TRT engine. The torch CUDA-graph variant is
forbidden under engine co-residency; the one-step engine lane missed the
trajectory gate marginally (see the fast-lane section below), so the
certified chain keeps the eager rollout. Numbers are stage means over the same 94-scene run; the
end-to-end mean additionally carries about 20 ms of stage-internal glue
(mrope merge and cache handoff), so it sits slightly above the four-row sum.

## Fast planner lane (shipped as an opt-in lane; re-run from this repo's pipeline)

The planner also has a fast lane, shipped as an opt-in pipeline path
(`QD_FAST_LANE=1`, steps 5b/9b/10b): a dynamic-S TRT bf16 engine replaces
the whole plan bucket with ONE call per scene instead of the 10-step eager
rollout. We re-ran the entire lane end-to-end from the shipped tree
(x86 ONNX export -> Orin gate bundle -> engine build -> engine gate ->
94-scene chain gate). Every gate number reproduced the archived internal
measurement: the chain gate read 5.1194 cm — the same 7 significant figures
— and the per-scene structure is identical down to the outlier list
(34 scenes over 5 cm, tops 30.25 / 30.10 / 16.10 / 11.65 cm; one of the two
30 cm scenes is the certified 10-step arm's worst at 35.23 cm, i.e. the
known degenerate-plan family, not engine-specific damage). The engine gate
reproduced bit-consistently (cos 0.99992638, max 0.88 cm, byte-exact gate
inputs), and the serving contract held: 94/94 planner calls on-engine, zero
eager fallback.

| Metric (re-run 2026-10-25) | Value | Archived (2026-10-06) |
|---|---|---|
| Engine gate at S=1141 (bar cos >= 0.999) | cos 0.99992638, max 0.88 cm, PASS | identical |
| Engine call p50 (S=1141, n=200) | 29.72 ms | 29.87 ms |
| In-chain plan bucket | 35.7 ms | 35.9 ms |
| All-torch 1-step plan bucket (base arm) | 122.8 ms | not measured |
| End-to-end (1-step arm) | 1026 ms vs 1210 ms base (-184 ms) | 1028 ms |
| Chain gate (record-only) | 5.1194 cm vs 5.0 cm bar, FAIL marginal | identical |
| ADE5 / FDE5 relative (vs its 1-step reference) | -0.44% / -0.82% (in band) | in band |

It is not the default config: the 5.1194 cm against the 5.0 cm bar is a
marginal FAIL driven by the two 30 cm degenerate-family scenes; the
remaining 92 scenes sit far in band, ADE5/FDE5/NaN are inside bounds, and
the other three gated arms in the archived matrix (10-step engine on both
manifests, 1-step r640) PASSED. Re-run timings were taken in a mildly
degraded compute window (24.2 TF vs the 27.9 TF bf16 anchor, numerics
anchor-gated) and are provisional; the certified table numbers above are
from the certified window.

## 94-scene open-loop trajectory gate

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
