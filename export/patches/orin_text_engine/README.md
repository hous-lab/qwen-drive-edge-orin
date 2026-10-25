# Edge-LLM patch bundle: sm_87 GDN chunk-prefill text engine

This bundle adapts NVIDIA TensorRT-Edge-LLM v0.9.1 for the qwen3_5_text prefill engine on Jetson AGX Orin (TRT 10.4, JetPack 6.x): it dispatches `seq_len > 1` GDN prefill to our sm_87 chunk-prefill kernel (shipped under [kernels/](../../../kernels/) at the repo root) instead of the official CuTe prefill path, which outputs all zeros on sm_87.

## Baseline

- Repo: `TensorRT-Edge-LLM` v0.9.1, git `7f061f2`. A stock v0.9.1 checkout targets TRT >= 10.5 (desktop): its `cpp/common/trtUtils.cpp` uses the `IStreamReaderV2` API that JetPack 6.x's TRT 10.4 python wheel does not carry — the first Orin build fails exactly there. The full TRT-10.4 compat delta (the `files/` overrides in the table below: stream-reader trtUtils, CUDA 12.6 CuTe-DSL shim, the two CMake overrides and build-config variants, the FP4-term drop, and the FMHA cubin-load hardening) is shipped in this bundle and staged by the same rsync step as the kernel files.
- `builder/llmBuilder.cpp`, `gatedDeltaNetPlugin`, `attentionPlugin` and `causalConvPlugin` are identical between the lineages, so the x86-exported ONNX ops are compatible with this base; the dispatch patch applies unchanged (the GDN runner `cuteDslGDNRunner.cpp` is identical pre-patch).
- Apply the bundle to a fresh tree per [build_plugin_orin.sh](build_plugin_orin.sh); keep vision-engine builds on their own tree and plugin .so.

## What's here

| file | role |
|---|---|
| [0001-gdn-sm87-chunk-prefill-dispatch.patch](0001-gdn-sm87-chunk-prefill-dispatch.patch) | Modifies `cpp/kernels/gdnKernels/cuteDslGDNRunner.cpp`: dispatch `seq_len > 1 && sm < 100` to `runGdnChunkPrefillSm87()` before the (sm_87-broken) official CuTe prefill. Decode (`seq_len==1`), MTP, DDTree paths untouched. |
| [files/cpp/kernels/gdnKernels/gdnChunkPrefillSm87.h](files/cpp/kernels/gdnKernels/gdnChunkPrefillSm87.h) / [.cu](files/cpp/kernels/gdnKernels/gdnChunkPrefillSm87.cu) | NEW files: fp16/bf16 adapters + gate math + q/k L2 norm around `deltanet_chunk_prefill` (the chunked delta-rule kernel shipped under [kernels/](../../../kernels/)), plus a growth-only static scratch shared by all plugin instances. |
| [files/cpp/kernels/posEncoding/applyRopeWriteKV.cu](files/cpp/kernels/posEncoding/applyRopeWriteKV.cu) | Overrides the base-tree copy: a debug printf block is wrapped in `#ifdef EDGE_GDN_WRITEKV_DEBUG` (stripped by default); re-enable per-build by adding `-DEDGE_GDN_WRITEKV_DEBUG` to the plugin target's compile flags. |
| `files/cpp/kernels/gatedRmsNorm/gatedRmsNormKernel.h/.cu` | NEW files: fused `y = w · RMSNorm(o) · silu(z)` kernel, fp16 I/O + fp32 math, one warp per (token, v-head) row. |
| `files/cpp/plugins/gatedRmsNormPlugin/gatedRmsNormPlugin.h/.cpp` | NEW files: IPluginV3 wrapper (`gated_rmsnorm`, TRT-10.4 API: `IPluginV3OneBuildV2`/`IPluginCreatorV3One`, eps float attr). Consumes GDN plugin O + per-head gate z + norm weight. Both auto-globbed into their CMake targets. |
| [files/cpp/common/trtUtils.cpp](files/cpp/common/trtUtils.cpp) | TRT-10.4 compat override: the stock `IStreamReaderV2` file-reader becomes `IStreamReader` (JetPack 6.x's TRT 10.4 wheel carries the V1 stream reader only); this is the exact API the first Orin build fails on. |
| [files/cpp/cuda126_cutedsl_compat.h](files/cpp/cuda126_cutedsl_compat.h) | NEW file: CUDA 12.6 shim for CuTe-DSL's 12.8+ runtime module API (`cudaLibrary_t` typedef + `cudaLibraryUnload` decl; definition linked from the tree's cudart shim). Force-included into kernels/plugin targets. |
| [files/cmake/CuteDsl.cmake](files/cmake/CuteDsl.cmake) | Override: links `trt_edgellm_cutedsl_cudart_shim` into CuTe-DSL targets (pairs with the compat header above). |
| [files/cmake/XQACubins.cmake](files/cmake/XQACubins.cmake) | Override: appends the native SM to XQA cubin generation, so decode XQA kernels match the running hardware even when `CMAKE_CUDA_ARCHITECTURES` omits it (Orin sm_87 vs the FMHA `80;86;89` set). |
| [files/cpp/CMakeLists.txt](files/cpp/CMakeLists.txt) | Override: the three `target_compile_options` that force-include [files/cpp/cuda126_cutedsl_compat.h](files/cpp/cuda126_cutedsl_compat.h) into edgellmKernels/edgellmCore/the plugin target, plus `plugins/*.cu` globbed into the plugin library (the gated_rmsnorm kernel needs this) and sm_87 in the FMHA SM list. Stock + 5 lines; without it the CuTe-DSL artifact headers fail on CUDA 12.6 (`cudaLibrary_t`/`cudaLibraryUnload` unknown). |
| [files/CMakeLists.txt](files/CMakeLists.txt) | Override (one line): `cublas` dropped from the `commonLibraryExt` INTERFACE link set. |
| [files/cpp/plugins/fusedNvfp4GemmAllReducePlugin/fusedNvfp4GemmAllReducePlugin.cpp](files/cpp/plugins/fusedNvfp4GemmAllReducePlugin/fusedNvfp4GemmAllReducePlugin.cpp) | TRT-10.4 compat: `nvinfer1::DataType::kFP4` does not exist before TRT 10.5 — both `supportsFormatCombination` predicates drop the FP4 term, so FP4 formats are never selected (matching what TRT 10.4 could build anyway). |
| [files/cpp/kernels/contextAttentionKernels/contextFMHARunner.cpp](files/cpp/kernels/contextAttentionKernels/contextFMHARunner.cpp) | Production-verified hardening override: stock code loads FMHA cubins with `CUDA_DRIVER_CHECK(cuModuleLoadData(...))` and **aborts** the process when a cubin fails to load for the running SM (two sm_87 FMHA cubins legitimately fail with driver error 200 on Orin at first use). The production copy loads a malloc'd copy via `cuModuleLoadDataEx` and skips the failed kernel with `LOG_WARNING` + `continue` (copies owned in `mOwnedCubins`). Without this override the clean-rebuilt plugin aborts at runtime, exactly as observed in production bring-up. |


Deltas deliberately shipped as stock: the production bring-up tree carried further local edits that this bundle intentionally does not reproduce — debug-print debris in embeddingKernels.cu, the unused event-centric explanation additions in llmInferenceRuntime, python-side int8dyn fusion hooks (never exercised by the fp16 production graph), and the alpamayo1ActionRunner/examples additions (out of scope here). A recursive diff of a staged tree against stock v0.9.1 therefore shows exactly the table above plus the dispatch patch and the two staged kernel files.

## Why

Official GDN prefill on sm_87 outputs ALL ZEROS (Prefill + PrefillPadding FAILED in the AOT kernel's own self-check; only decode passes; the Blackwell-prefill variant is gated SM >= 100). Kernel numerics on sm_87 (state track, fp32): rel ~1e-5 vs the fp32 reference with cos 1.000000 on real-activation fixtures at T=3385; the bf16 output track pins tensor L2 rel at ~2-3e-3 (half a bf16 ulp), validated at model level (KV cos >= 0.9996, trajectory cos 0.99997) and by the 94-scene open-loop trajectory gate. The production-contract parity arm (q/k L2-normalized) passes the blanket parity bar 6/6 (rel <= 9.8e-3, cos >= 0.99991); the raw (non-normalized) arm sits at 1.2-1.5e-2 at B=2 in the production records on both x86 and Orin — the same signature the shipped harness reproduces. Dispatch sits in `CuteDslGDNRunner::run()` so one binary serves both paths.

## Contract notes (gdnChunkPrefillSm87)

- Exact-T: every batch item valid length == `seq_len` (engine executes prefill with the true token count; production batch = 1, T = 640/1141). `context_lengths` is not consulted.
- Gate math: `g = -exp(A_log)·softplus(a + dt_bias)` fp32, `beta = sigmoid(b)` fp32; q/k L2-normalized fp32 eps 1e-6 then cast to bf16.
- h0: plugin has already D2D-copied h0_source→h0_out before dispatch, so `params.h0_source` is the working state buffer: kernel reads initial state from it, writes final state to scratch, async-copies back (no aliasing assumptions).
- Scratch: one growth-only `cudaMalloc`, shared by all 24 plugin instances (sequential within one enqueue). ~100 MB class at T=640, B=1.
- M-RoPE feeding: prefill table row t = composite mrope row of token t ([runtime/chain/mrope_rows.py](../../../runtime/chain/mrope_rows.py), kv_start=0 → kernel reads row = token index). Pad rows >= S are arbitrary (arange) — never read. Decode tokens are text (t=h=w=position), so the standard arange row is already correct. One hard lesson from the bring-up: a numpy transpose view (F-contiguous array) poisons every downstream buffer built from it while staying invisible to logical-content checks — always assert contiguity on tensors handed to raw-pointer consumers (TRT engines, CUDA kernels).

## gated_rmsnorm plugin status

The plugin target compiles the gated_rmsnorm plugin+kernel (auto-globbed), but production keeps the Myelin-compiled norm+silu+mul chain: a paired A/B measured net zero at both tested shapes (T=1503 and the production T=1141, both inside run noise), while kernel parity itself was 99.9953% bitwise against the fp16 HF-reference math. Shipped as a dormant capability, not a dispatched node in the production graph. Two build traps worth knowing:

- `llm_build` defaults `--maxInputLen 1024 --maxBatchSize 4`; the production engine is built with `--maxBatchSize 1 --maxInputLen 4096 --maxKVCacheCapacity 4096` or T=1141 feeds are rejected by the optimization profile.
- Export `EDGELLM_PLUGIN_PATH=$QD_PLUGIN_SO` BEFORE `llm_build` (ONNX parse needs the registered plugin creators).

## Kill switch

`QD_GDN_OFFICIAL=1` restores the official (broken-on-sm87) prefill path — read once via `getenv` at first dispatch, so set it per-process. This is the baseline-reproduction switch for the GDN prefill-zeros defect and for A/B.

## Apply + build on Orin

Run the shipped build script on the Orin host:

    bash export/patches/orin_text_engine/build_plugin_orin.sh

Steps inside: stage [files/](files/) over the checkout → apply the dispatch patch → cmake configure (Release / CUDA 12.6 / ARCH 87) → build `NvInfer_edgellm_plugin`. Runtime needs the plugin on the loader path: `EDGELLM_PLUGIN_PATH=$QD_PLUGIN_SO`.
