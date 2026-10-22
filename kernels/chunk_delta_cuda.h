// CUDA C++ port of harness/chunk_delta.py
// (self-written chunked gated delta rule, G0-passed Triton version) for the E3
// text-prefill TRT plugin. Plain CUDA C API: no torch / TRT dependency here so
// the plugin can link this file directly.
//
// Math is term-by-term aligned with transformers 5.14.1
// modeling_qwen3_5.py::torch_chunk_gated_delta_rule (L248); see the Triton
// file's header for the full derivation and the three-kernel (parallel prep +
// serial state carry + parallel per-chunk output pass) topology rationale.
//
// IO contract (contiguous, stream-ordered):
//   q, k : bf16 [B, T, HQK, DK]  -- HQK may be HV (parity fixtures) or the
//          physical key-head count (e.g. 16); HV head h reads qk head
//          h / (HV/HQK), i.e. repeat_interleave semantics.
//   v    : bf16 [B, T, HV, DV]
//   g    : fp32 [B, T, HV]   (negative log-decay, cumsumed inside chunks)
//   beta : fp32 [B, T, HV]
//   o    : bf16 [B, T, HV, DV]                (out)
//   s0   : fp32 [B, HV, DK, DV] or NULL (=> zeros)
//   sf   : fp32 [B, HV, DK, DV]               (out)
//
// Precision: all dots are plain fp32 FMA (>= the shipped Triton "tf32x3"
// delivery config); transcendentals via expf. L2-norm of q/k is NOT done here
// -- the caller normalizes (as the model does when use_qk_l2norm_in_kernel).
//
// Scratch: caller-provided workspace (see deltanet_prefill_workspace_size);
// TRT plugin can map this to its own workspace later.
#ifndef DELTANET_CHUNK_DELTA_CUDA_H_
#define DELTANET_CHUNK_DELTA_CUDA_H_

#include <cuda_runtime.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Workspace bytes needed for one prefill call with these dims.
size_t deltanet_prefill_workspace_size(int B, int T, int HV, int DK, int DV, int chunk_size);

// Chunked gated delta rule prefill. DK=DV=128 and chunk_size=64 are the only
// validated configuration (kernel templates are instantiated for it).
// dump_debug != 0 additionally writes per-chunk T/A/post-leaf-T (each
// [B,HV,NC,chunk,chunk] fp32) past the live scratch at the tail of the
// workspace (debug harnesses only; costs extra global bandwidth when set).
// Returns cudaSuccess or the CUDA error encountered.
cudaError_t deltanet_chunk_prefill(
    const void* q, const void* k, const void* v,  // bf16
    const float* g, const float* beta,            // fp32
    void* o,                                      // bf16 out
    const float* initial_state,                   // fp32 or NULL
    float* final_state,                           // fp32 out
    int B, int T, int HV, int HQK, int DK, int DV, int chunk_size,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream, int dump_debug);

#ifdef __cplusplus
}
#endif

#endif  // DELTANET_CHUNK_DELTA_CUDA_H_
