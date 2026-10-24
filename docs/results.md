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
| Planner | torch eager (PlanningExpert, one-step) | 1162.7 ms |
| End to end | vision + text + planner | 2191.5 ms |

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
