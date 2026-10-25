# qwen-drive-edge-orin

Deployment artifacts and evaluation harness for running the [Qwen-Drive-1.0-4B](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B) VLA on Jetson AGX Orin as a TensorRT pipeline: a self-developed sm_87 chunk-prefill kernel for the GDN layers inside NVIDIA TensorRT-Edge-LLM, a TRT fp16 vision engine, torch bf16 text prefill, and a one-step TRT bf16 planner engine (one call per scene). Text runs eagerly in the production chain because the text engine and the planner engine cannot share one process (engine co-residency).

The production chain runs at 1026 ms per step end to end (re-run from this repo's pipeline, 2026-10-25; planner stage 1162.7 ms in the archived 10-step baseline). Accuracy is gated on ground-truth-anchored task metrics (ADE5/FDE5 within +10% of the fp 10-step reference semantics) plus the engine-serving contract; the plan-space twin-fidelity record is in [docs/results.md](docs/results.md). Closed-loop evaluation is documented as a pointer only [ckpts/README.md](ckpts/README.md).

## Results

| Stage | Implementation | Latency (Orin, re-run 2026-10-25 window, provisional) |
|---|---|---|
| Process (tokenize + glue) | torch | 208.2 ms |
| Prefill (vision + text) | vision TRT fp16 plan + text torch bf16 eager | 782.0 ms |
| Planner | TRT bf16 one-step engine, dynamic-S | 35.7 ms |
| End to end | full chain | 1026.0 ms |

Archived certified baseline (2026-10-12 certified window): vision TRT 211.6 ms + text TRT engine 580.5 ms + planner eager 10-step 1162.7 ms + orchestration 216.6 ms = 2191.5 ms end to end.

The planner serves every scene with ONE engine call per plan (35.7 ms in-chain; 29.7 ms standalone p50 at S=1141). Task-level comparison over the 94-scene manifest, ground-truth-anchored:

| Arm | ADE5 (m) | FDE5 (m) | vs 10-step fp reference | Plan latency |
|---|---|---|---|---|
| 10-step torch (fp semantics reference) | 2.589 | 6.894 | — | 1162.7 ms |
| 1-step torch | 2.665 | 7.223 | +2.93% / +4.78% | 122.8 ms |
| 1-step engine (production) | 2.653 | 7.164 | **+2.48% / +3.92%** | 35.7 ms |

The engine adds no measurable task damage — it is marginally closer to ground truth vs its own 1-step torch twin (47/94 scenes better, mean ADE5 change −1.2 cm); the +2.48/+3.92% is the one-step sampler semantics itself, which a pure-torch 1-step arm shows identically, and it sits far inside the ±10% band. The chain-level twin-fidelity metric (engine arm vs its same-semantics torch reference) reads 5.1194 cm mean abs plan delta; [docs/results.md](docs/results.md) explains why this co-exists with the task metrics above — twin fidelity is a change detector, not a quality metric, and it is more sensitive in the one-step regime.

Production chain gate (one-step engine planner arm, eval_chain_v3 over the 94-scene manifest):

| Metric | Value | Criterion |
|---|---|---|
| ADE5 / FDE5 vs ground truth, relative to the 10-step fp reference | +2.48% / +3.92% | <= +10% |
| NaN scenes | 0 | 0 allowed |
| Engine-serving contract (planner calls on-engine) | 94/94 | 94/94 required (eager fallback is FATAL) |
| Plan-space twin fidelity (engine arm vs 1-step torch twin) | 5.1194 cm mean abs | record-only, see [docs/results.md](docs/results.md) |

The archived 10-step gate (engine text prefill arm, 1.3076 cm mean abs trajectory delta PASS) remains reproducible with `QD_FAST_LANE=0`.

## Takeaways

- The GDN hybrid layers, not the attention layers, are the text-prefill bottleneck on sm_87; a chunk-parallel prefill kernel with bf16 wmma in the output segment moves the stage by -22% and end-to-end by -144.5 ms.
- Task-level gates can pass while a numerical-parity lane still disagrees — keep both, gate on the task metric, and read [docs/results.md](docs/results.md) before over-claiming parity.
- The production planner is a one-step TRT bf16 engine (35.7 ms in-chain vs 1162.7 ms for the eager 10-step rollout); its task metrics sit within +2.5%/+3.9% of the fp 10-step reference semantics, and INT8 routes stay documented as closed with root causes in [docs/negative_results.md](docs/negative_results.md).

## Get started

1. Clone this repo and the upstream qwen-drive repo side by side, the two-repo workspace contract in [scripts/env_orin.sh](scripts/env_orin.sh) OR [scripts/env_x86.sh](scripts/env_x86.sh).
2. `./scripts/fetch_upstream.sh` after filling a copy of [scripts/env_x86.sh](scripts/env_x86.sh) — pulls the model weights, clones the upstream code, and prints the gated [PAI dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) steps.
3. `./run_pipeline.sh <env.sh> x86` — data adapter, vision ONNX export + gate, text gate reference.
4. On Orin: `./run_pipeline.sh <env.sh> orin` — plugin build, kernel parity gate, chain gate.

## Hardware and limits

- Certified on Jetson AGX Orin 64GB (sm_87), JetPack 6.x; the kernel is sm_87-specific (wmma tile shapes, smem budget).
- x86 export host: any CUDA 12.8-capable GPU with >= 16 GB; the vision ONNX export loads the full 4B model.
- The TRT 10.4 compat delta for the Edge-LLM build is a known release gap [docs/environment.md](docs/environment.md).

## Repository map

- [export/](export/) — x86 exporters and gates (vision ONNX, text reference bundle)
- [kernels/](kernels/) — the sm_87 GDN chunk-prefill kernel, parity harness, and the production-integration notes [kernels/README.md](kernels/README.md)
- [runtime/chain/](runtime/chain/) — chain runner + evaluation gates
- [eval/adapter/](eval/adapter/) — PAI dataset adapter
- [export/patches/orin_text_engine/](export/patches/orin_text_engine/) — Edge-LLM patch bundle + build script
- [configs/](configs/) — certified build flags and the f2_05 eval manifest
- [docs/](docs/) — environment, pipeline guide, results, negative results
- [ckpts/README.md](ckpts/README.md) — upstream artifact policy + download table

## Acknowledgements and license

Apache-2.0. Built on the [Qwen-Drive-1.0-4B](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B) model (Apache-2.0) and NVIDIA [TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM).
PAI driving data via the gated NVIDIA [PhysicalAI AV dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).
This is an independent engineering effort, not affiliated with Qwen, NVIDIA, or the dataset authors — see [NOTICE](NOTICE) for the full attribution list.
