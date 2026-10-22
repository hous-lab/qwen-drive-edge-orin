# qwen-drive-edge-orin

Deployment artifacts and evaluation harness for running the Qwen-Drive-1.0
4B VLA on Jetson AGX Orin as a TensorRT pipeline: a self-developed sm_87
chunk-prefill kernel for the GDN layers inside NVIDIA TensorRT-Edge-LLM,
fp16 engines for vision and text prefill, and the torch planner on top.

The pipeline passes a 94-scene open-loop trajectory gate (engine arm vs
torch reference arm, mean abs delta 1.31 cm) on Orin, with the engine chain
at 2191.5 ms per step. Closed-loop evaluation is documented as a pointer
only (ckpts/README.md).

## Results

| Stage | Implementation | Latency (Orin, locked clocks) |
|---|---|---|
| Vision | TRT fp16 plan | 211.6 ms |
| Text prefill | TRT fp16 engine, self-developed GDN sm_87 kernel | 580.5 ms (T=1141) |
| Planner | torch eager (one-step) | 1162.7 ms |
| End to end | full chain | 2191.5 ms |

94-scene trajectory gate (engine arm vs torch reference arm,
eval_chain_v3 aggregate-blocking):

| Metric | Value | Criterion |
|---|---|---|
| Mean abs trajectory delta | 1.3076 cm | <= 5 cm |
| ADE5 / FDE5 relative | in band | <= +10% |
| NaN scenes | 0 | 0 allowed |

## Takeaways

- The GDN hybrid layers, not the attention layers, are the text-prefill
  bottleneck on sm_87; a chunk-parallel prefill kernel with bf16 wmma in the
  output segment moves the stage by -22% and end-to-end by -144.5 ms.
- Task-level gates can pass while a numerical-parity lane still disagrees —
  keep both, gate on the task metric, and read docs/results.md before
  over-claiming parity.
- The fp16 production chain is the quality-latency sweet spot we certified;
  INT8 routes are documented as closed with root causes in
  docs/negative_results.md.

## Get started

1. Clone this repo and the upstream qwen-drive repo side by side (the
   two-repo workspace contract in scripts/env_*.sh).
2. `./scripts/fetch_upstream.sh` after filling a copy of `scripts/env_x86.sh`
   — pulls the model weights, clones the upstream code, and prints the gated
   PAI dataset steps.
3. `./run_pipeline.sh <env.sh> x86` — data adapter, vision ONNX export +
   gate, text gate reference.
4. On Orin: `./run_pipeline.sh <env.sh> orin` — plugin build, kernel parity
   gate, chain gate.

## Hardware and limits

- Certified on Jetson AGX Orin 64GB (sm_87), JetPack 6.x; the kernel is
  sm_87-specific (wmma tile shapes, smem budget).
- x86 export host: any CUDA 12.8-capable GPU with >= 16 GB; the vision ONNX
  export loads the full 4B model.
- The TRT 10.4 compat delta for the Edge-LLM build is a known release gap
  (docs/environment.md).

## Repository map

- `export/` — x86 exporters and gates (vision ONNX, text reference bundle)
- `kernels/` — the sm_87 GDN chunk-prefill kernel, parity harness, and the
  production-integration notes (kernels/README.md)
- `runtime/chain/` — chain runner + evaluation gates
- `eval/adapter/` — PAI dataset adapter
- `export/patches/orin_text_engine/` — Edge-LLM patch bundle + build script
- `configs/` — certified build flags and the f2_05 eval manifest
- `docs/` — environment, pipeline guide, results, negative results
- `ckpts/README.md` — upstream artifact policy + download table

## Acknowledgements and license

Apache-2.0. Built on the Qwen-Drive-1.0-4B model (Apache-2.0) and NVIDIA
TensorRT-Edge-LLM; PAI driving data via the gated NVIDIA PhysicalAI AV
dataset. This is an independent engineering effort, not affiliated with
Qwen, NVIDIA, or the dataset authors — see NOTICE for the full attribution
list.
