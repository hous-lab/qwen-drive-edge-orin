# Release notes — v1.0

Initial versioned release of the Qwen-Drive-1.0-4B Jetson AGX Orin deployment: a TRT fp16 vision engine, bf16 eager text prefill, and a one-step TRT bf16 planner engine (one engine call per trajectory), plus the full gate harness that produced the published numbers.

## Tested device and dependency versions

| Component | Version |
|---|---|
| Device | Jetson AGX Orin 64GB (sm_87), JetPack 6.2.2 / L4T R36.5.0, MAXN + `jetson_clocks` |
| TensorRT | 10.4 |
| TensorRT-Edge-LLM | v0.9.1 + the shipped patch bundle ([export/patches/orin_text_engine/](export/patches/orin_text_engine/)) |
| torch | 2.8.0 |
| Export host | x86, CUDA 12.8-capable GPU (>= 16 GB) |

## Headline metrics

All latencies are means over one 94-scene open-loop pass, single stream, batch 1 (measurement metadata and full evidence chain: [docs/results.md](docs/results.md)).

| Metric | Value |
|---|---|
| Production chain end to end | 1026.0 ms (process 208.2 + prefill 782.0 + planner 35.7) |
| Planner engine standalone | 29.7 ms p50 (200 timed calls after 20 warmup, S=1141) |
| Baseline chain end to end (`QD_FAST_LANE=0`) | 2191.5 ms, all gates PASS at 1.3076 cm |
| Task accuracy vs 10-step torch reference | ADE5 +2.48% / FDE5 +3.92% (gate band <= +10%), one-step sampler semantics |
| Engine vs its own 1-step torch twin | marginally closer to ground truth (47/94 scenes better) |
| Engine-serving contract | 94/94 scenes on-engine (eager fallback FATAL) |

## Known issues

- The text engine and the planner engine cannot share one process: multi-engine co-residency corrupts small tensors (heisenbug, per-engine clean). The production chain runs text prefill eagerly in bf16 for this reason; see [docs/negative_results.md](docs/negative_results.md).
- The shipped sm_87 GDN kernel is exercised by the baseline text-engine lane (`QD_FAST_LANE=0`) and is sm_87-specific (wmma tile shapes, shared-memory budget).
- A minority of manifest scenes produce degenerate plans in all arms (including the 10-step torch reference); the gate tolerates them via the aggregate criteria. Twin-fidelity outliers concentrate there — see the twin-fidelity analysis in [docs/results.md](docs/results.md).
- One internal numerical-parity lane reads kv-cos ~0.88 while the engine gate reads 0.99993 and task metrics are in band; UNRESOLVED, with checkpoint forensics pointing at gate comparison semantics rather than demonstrated numerical corruption. Documented in [docs/results.md](docs/results.md) with FACT/INTERPRETATION/HYPOTHESIS separation.
- Latencies were measured on a unit whose compute window runs mildly degraded (24.2 TF vs the 27.9 TF bf16 anchor); treat absolute values as unit-specific — A/B deltas and gate verdicts are anchor-gated and cross-run stable to 7 significant figures.

## Incompatible changes

None — initial versioned release. The `QD_FAST_LANE` switch selects between the production one-step lane (default) and the baseline 10-step lane.

## Traceability

Every number in these notes and in [docs/results.md](docs/results.md) regenerates from this tag via the repo pipeline: `./run_pipeline.sh <env.sh> x86` then `./run_pipeline.sh <env.sh> orin` (upstream artifacts per [ckpts/README.md](ckpts/README.md)). Gate summaries are archived by the pipeline under the runtime logs directory.
