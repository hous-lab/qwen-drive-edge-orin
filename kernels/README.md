# kernels — GDN sm_87 chunk-prefill CUDA kernel

Self-developed CUDA kernel for the Qwen3.5 **GDN (gated delta rule) chunk prefill** executed inside the text prefill engine on Jetson AGX Orin (sm_87). NVIDIA's official CuTe GDN prefill kernel produces all-zero output on sm_87 (see [export/patches/orin_text_engine/README.md](../export/patches/orin_text_engine/README.md)), so the engine dispatches to this kernel instead. The text engine built by this repo's recipe runs the **bf16-wmma o-pass variant**, shipped as [chunk_delta_cuda_tco.cu](chunk_delta_cuda_tco.cu).

## Files

| File | What it is | md5 (shipped) |
|---|---|---|
| [chunk_delta_cuda.cu](chunk_delta_cuda.cu) / [chunk_delta_cuda.h](chunk_delta_cuda.h) | Canonical kernel (fp32 tensor-op path), entry `deltanet_chunk_prefill` — v1-era baseline, and the API the TRT plugin wrapper calls | `a065f1bf` / `a553bffa` |
| [chunk_delta_cuda_tco.cu](chunk_delta_cuda_tco.cu) / [chunk_delta_cuda_tco.h](chunk_delta_cuda_tco.h) | **bf16-wmma o-pass variant (production)**: o-pass rewritten as all-bf16 wmma with fp32 accumulate, entry `deltanet_tcop_chunk_prefill` | `5283bd48` / `bdad0f97` |
| [tco_body.txt](tco_body.txt) | Exact o-kernel body swapped into the production plugin `.so` via objcopy re-link (see below) | `90dc73fb` |
| [torch_binding.cpp](torch_binding.cpp) / [torch_binding_tco.cpp](torch_binding_tco.cpp) | Minimal pybind wrappers (torch extension) used by the standalone parity harness | — |
| [parity_tco.py](parity_tco.py), [harness/](harness/) | Standalone parity + latency harness: builds the kernel as a torch extension and compares vs the torch fallback reference and vs the self-written Triton kernel ([harness/chunk_delta.py](harness/chunk_delta.py)) | — |

md5 anchors for the production artifacts (build outputs, not shipped): production plugin `libNvInfer_edgellm_plugin_tcoexp.so.1.0` md5 `dc3eb745…`; the shipped [chunk_delta_cuda.cu](chunk_delta_cuda.cu) is the source anchor (md5 `a065f1bf`) recorded in the production baseline notes.

## Standalone parity gate

From [kernels/](.) (no TRT required — compiles the kernel as a torch extension):

```bash
# Orin: TORCH_CUDA_ARCH_LIST=8.7 python parity_tco.py
# x86:  QD_CUDA_HOME=/usr/local/cuda-12.4 python parity_tco.py
```

Pass bar (per case, on `o` and `final_state`): min per-row cosine ≥ 0.999 and max relative error < 1e-2. The reference is the pure-torch fallback of the upstream transformers implementation; [harness/chunk_delta.py](harness/chunk_delta.py) is a self-written Triton implementation aligned term-by-term with the upstream math, shipped as a readable reference and x86-viable baseline.

Known signature, so a FAIL there is not a regression: the raw (non-L2-normalized) arm — the kernel-contract path, not what the plugin runs — sits at rel 1.2–1.5e-2 at B=2 (above the blanket bar) in the production records on both x86 and Orin; the production-contract arm (q/k L2-normalized, what the plugin wrapper always runs) passes the blanket bar 6/6. The per-layer `o` track is bf16 output: expect tensor L2 rel ~2–3e-3 (half a bf16 ulp) and the fp32 `final_state` track rel ~1e-5–1e-6. Real-activation fixture rows need the fixtures under `harness/deltanet_kernel/fixtures/` (not shipped). The production verdict came from the model-level gates (KV cos ≥ 0.9996 / trajectory cos 0.99997) and the 94-scene open-loop trajectory gate, not this harness's blanket verdict.

## Production integration (TRT plugin)

The engine consumes the kernel through the TRT plugin `gdnChunkPrefillSm87` (wrapper source in [export/patches/orin_text_engine/files/cpp/kernels/gdnKernels/](../export/patches/orin_text_engine/files/cpp/kernels/gdnKernels/)), which adapts bf16 q/k/v + fp32 g/beta to the kernel contract and calls `deltanet_chunk_prefill`. Build path: apply [export/patches/orin_text_engine/0001-gdn-sm87-chunk-prefill-dispatch.patch](../export/patches/orin_text_engine/0001-gdn-sm87-chunk-prefill-dispatch.patch) to a fresh TensorRT-Edge-LLM v0.9.1 checkout, stage `chunk_delta_cuda.{cu,h}` (canonical names), and build with [export/patches/orin_text_engine/build_plugin_orin.sh](../export/patches/orin_text_engine/build_plugin_orin.sh).

**Deployment vehicle for the bf16-wmma o-pass variant, stated precisely:** the wrapper's declared API is the canonical entry `deltanet_chunk_prefill`. The production plugin (md5 `dc3eb745…`, artifact name `libNvInfer_edgellm_plugin_tcoexp.so.1.0`) was produced by compiling the variant's o-kernel body ([tco_body.txt](tco_body.txt)) and re-linking it into the plugin via objcopy, replacing the canonical o-pass body inside `deltanet_chunk_prefill`'s call graph. The variant's standalone entry `deltanet_tcop_chunk_prefill` in [chunk_delta_cuda_tco.cu](chunk_delta_cuda_tco.cu) is the same math as a self-contained kernel; it is what the parity harness builds and gates. A clean rebuild that stages [chunk_delta_cuda_tco.cu](chunk_delta_cuda_tco.cu) under the canonical name additionally requires pointing the wrapper at `deltanet_tcop_chunk_prefill` + `deltanet_tcop_prefill_workspace_size` (workspace layout differs) — this path is **not** the shipped vehicle and is not exercised by the shipped recipe; if you rebuild from source, the shipped vehicle is the canonical build from [chunk_delta_cuda.cu](chunk_delta_cuda.cu) (source anchor md5 `a065f1bf`), and the production `.so` is the swap artifact documented above.

`QD_GDN_OFFICIAL=1` restores NVIDIA's official CuTe GDN kernel at runtime (all-zero output on sm_87 — kept as the documented fallback toggle from the dispatch patch).

## Results (device: Jetson AGX Orin, locked clocks)

O-segment speedup vs canonical kernel: ×2.16–4.32 (two measurement protocols); end-to-end text engine segment −166.7 ms (−22%) at T=1141; full-chain e2e −144.5 ms. Trajectory gate on the 94-scene manifest (text-engine lane, `QD_FAST_LANE=0`): 1.3076 cm mean |Δ|, PASS (eval_chain_v3, aggregate-blocking). Full numbers and environment fingerprint in [docs/results.md](../docs/results.md).
