/*
 * E3 Qwen-Drive text engine: sm_87 GDN prefill via the self-developed chunked
 * delta-rule kernel (the chunk_delta_cuda.{cu,h} shipped under kernels/ in
 * the qwen-drive-edge-orin repo, bf16 wmma), bridged to the plugin's fp16 I/O
 * with conversion + gate + L2-norm adapters.  The official CuTe prefill
 * kernel emits all zeros on sm_87 (decode is fine) — see the patches README.
 */
#pragma once

#include "cuteDslGDNRunner.h"

namespace trt_edgellm
{
/**
 * Run one GDN prefill (seq_len > 1) with our chunk kernel.
 *
 * Contract (exact-T, B=1 production): every batch item has valid length
 * == params.seq_len; params.context_lengths is not consulted.  Returns 0 on
 * success, non-zero (and logs) on failure.
 */
int runGdnChunkPrefillSm87(GDNParams const& params, cudaStream_t stream);
} // namespace trt_edgellm
