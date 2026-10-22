# Edge-LLM patch bundle: sm_87 GDN chunk-prefill text engine

This bundle adapts NVIDIA TensorRT-Edge-LLM v0.9.1 for the qwen3_5_text
prefill engine on Jetson AGX Orin (TRT 10.4, JetPack 6.x): it dispatches
`seq_len > 1` GDN prefill to our sm_87 chunk-prefill kernel (shipped under
`kernels/` at the repo root) instead of the official CuTe prefill path,
which outputs all zeros on sm_87.

## Baseline

- Repo: `TensorRT-Edge-LLM` v0.9.1, git `7f061f2`. A stock v0.9.1 checkout
  targets TRT >= 10.5 (desktop): its `cpp/common/trtUtils.cpp` uses the
  `IStreamReaderV2` API that JetPack 6.x's TRT 10.4 python wheel does not
  carry — the first Orin build fails exactly there. The production lineage
  trees carry the TRT-10.4 compat deltas (`trtUtils.cpp` stream-API variant,
  `cpp/cuda126_cutedsl_compat.h` force-included into kernels/plugin targets,
  and Orin-proven `cmake/` fragments); extraction into `export/patches/` is
  the documented known gap — see docs/environment.md.
- `builder/llmBuilder.cpp`, `gatedDeltaNetPlugin`, `attentionPlugin` and
  `causalConvPlugin` are identical between the lineages, so the x86-exported
  ONNX ops are compatible with this base; the dispatch patch applies
  unchanged (the GDN runner `cuteDslGDNRunner.cpp` is identical pre-patch).
- Apply the bundle to a fresh tree per `build_plugin_orin.sh`; keep
  vision-engine builds on their own tree and plugin .so.

## What's here

| file | role |
|---|---|
| `0001-gdn-sm87-chunk-prefill-dispatch.patch` | Modifies `cpp/kernels/gdnKernels/cuteDslGDNRunner.cpp`: dispatch `seq_len > 1 && sm < 100` to `runGdnChunkPrefillSm87()` before the (sm_87-broken) official CuTe prefill. Decode (`seq_len==1`), MTP, DDTree paths untouched. |
| `files/cpp/kernels/gdnKernels/gdnChunkPrefillSm87.h/.cu` | NEW files: fp16/bf16 adapters + gate math + q/k L2 norm around `deltanet_chunk_prefill` (the chunked delta-rule kernel shipped under `kernels/`), plus a growth-only static scratch shared by all plugin instances. |
| `files/cpp/kernels/posEncoding/applyRopeWriteKV.cu` | Overrides the base-tree copy: a debug printf block is wrapped in `#ifdef EDGE_GDN_WRITEKV_DEBUG` (stripped by default); re-enable per-build by adding `-DEDGE_GDN_WRITEKV_DEBUG` to the plugin target's compile flags. |
| `files/cpp/kernels/gatedRmsNorm/gatedRmsNormKernel.h/.cu` | NEW files: fused `y = w · RMSNorm(o) · silu(z)` kernel, fp16 I/O + fp32 math, one warp per (token, v-head) row. |
| `files/cpp/plugins/gatedRmsNormPlugin/gatedRmsNormPlugin.h/.cpp` | NEW files: IPluginV3 wrapper (`gated_rmsnorm`, TRT-10.4 API: `IPluginV3OneBuildV2`/`IPluginCreatorV3One`, eps float attr). Consumes GDN plugin O + per-head gate z + norm weight. Both auto-globbed into their CMake targets. |

## Why

Official GDN prefill on sm_87 outputs ALL ZEROS (Prefill + PrefillPadding
FAILED in the AOT kernel's own self-check; only decode passes; the
Blackwell-prefill variant is gated SM >= 100). Our kernel passed the
real-activation fixture gates (output cos >= 0.9999 / state rel < 1e-3,
layer00/layer30) on sm_87. Dispatch sits in `CuteDslGDNRunner::run()` so one
binary serves both paths.

## Contract notes (gdnChunkPrefillSm87)

- Exact-T: every batch item valid length == `seq_len` (engine executes prefill
  with the true token count; production batch = 1, T = 640/1141).
  `context_lengths` is not consulted.
- Gate math: `g = -exp(A_log)·softplus(a + dt_bias)` fp32, `beta = sigmoid(b)`
  fp32; q/k L2-normalized fp32 eps 1e-6 then cast to bf16.
- h0: plugin has already D2D-copied h0_source→h0_out before dispatch, so
  `params.h0_source` is the working state buffer: kernel reads initial state
  from it, writes final state to scratch, async-copies back (no aliasing
  assumptions).
- Scratch: one growth-only `cudaMalloc`, shared by all 24 plugin instances
  (sequential within one enqueue). ~100 MB class at T=640, B=1.
- M-RoPE feeding: prefill table row t = composite mrope row of token t
  (`runtime/chain/mrope_rows.py`, kv_start=0 → kernel reads row = token
  index). Pad rows >= S are arbitrary (arange) — never read. Decode tokens
  are text (t=h=w=position), so the standard arange row is already correct.
  One hard lesson from the bring-up: a numpy transpose view (F-contiguous
  array) poisons every downstream buffer built from it while staying
  invisible to logical-content checks — always assert contiguity on tensors
  handed to raw-pointer consumers (TRT engines, CUDA kernels).

## gated_rmsnorm (E-GRN) status

The plugin target compiles the gated_rmsnorm plugin+kernel (auto-globbed),
but production keeps the Myelin-compiled norm+silu+mul chain: a paired A/B
measured net zero at both tested shapes (T=1503 and the production T=1141,
both inside run noise), while kernel parity itself was 99.9953% bitwise
against the fp16 HF-reference math. Shipped as a dormant capability, not a
dispatched node in the production graph. Two build traps worth knowing:

- `llm_build` defaults `--maxInputLen 1024 --maxBatchSize 4`; the production
  engine is built with `--maxBatchSize 1 --maxInputLen 4096
  --maxKVCacheCapacity 4096` or T=1141 feeds are rejected by the optimization
  profile.
- Export `EDGELLM_PLUGIN_PATH=$QD_PLUGIN_SO` BEFORE `llm_build` (ONNX parse
  needs the registered plugin creators).

## Kill switch

`QD_GDN_OFFICIAL=1` restores the official (broken-on-sm87) prefill path —
read once via `getenv` at first dispatch, so set it per-process. This is the
baseline-reproduction switch for the GDN prefill-zeros defect and for A/B.

## Apply + build on Orin

Run the shipped build script on the Orin host:

    bash export/patches/orin_text_engine/build_plugin_orin.sh

Steps inside: stage `files/` over the checkout → apply the dispatch patch →
cmake configure (Release / CUDA 12.6 / ARCH 87) → build
`NvInfer_edgellm_plugin`. Runtime needs the plugin on the loader path:
`EDGELLM_PLUGIN_PATH=$QD_PLUGIN_SO`.
