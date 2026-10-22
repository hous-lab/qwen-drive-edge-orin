/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * GatedRmsNorm kernel: y = w * round(RMSNorm(o)) * silu(z), FP16 I/O, fp32 math.
 * One warp per row (row = one (token, v-head) pair); verified 99.9953% bitwise
 * / cos 0.99999994 vs the fp16 HF-reference math on Orin (rows=1503*32,
 * dv=128) — a paired A/B showed net-zero wall-clock, so production keeps the
 * Myelin-compiled norm+silu+mul chain (see the patches README).
 */
#include "gatedRmsNormKernel.h"

#include <cuda_fp16.h>

namespace kernel
{

namespace
{

__global__ void kern_gated_rmsnorm_fp16(const __half* __restrict__ o, const __half* __restrict__ z,
    const __half* __restrict__ w, __half* __restrict__ y, int64_t rows, int dv, float eps)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t row = (int64_t) blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows)
    {
        return;
    }
    const __half* orow = o + row * dv;
    const __half* zrow = z + row * dv;
    float ss = 0.f;
    for (int j = lane; j < dv; j += 32)
    {
        const float v = __half2float(orow[j]);
        ss = fmaf(v, v, ss);
    }
#pragma unroll
    for (int off = 16; off; off >>= 1)
    {
        ss += __shfl_xor_sync(0xffffffffu, ss, off);
    }
    const float rms = rsqrtf(ss / (float) dv + eps);
    for (int j = lane; j < dv; j += 32)
    {
        const float n = __half2float(orow[j]) * rms;
        const float nb = __half2float(__float2half(n)); // reference .to(input_dtype)
        const float zf = __half2float(zrow[j]);
        const float s = zf * (1.f / (1.f + expf(-zf)));  // silu(z) in fp32
        const float wf = __half2float(w[j]);
        y[row * dv + j] = __float2half((wf * nb) * s);
    }
}

} // namespace

void gatedRmsNormForward(void const* o, void const* z, void const* w, void* y, int64_t rows, int dv, float eps,
    cudaStream_t stream)
{
    const int threads = 256; // 8 warps = 8 rows per block
    const int64_t blocks = (rows + (threads / 32) - 1) / (threads / 32);
    kern_gated_rmsnorm_fp16<<<(int) blocks, threads, 0, stream>>>(static_cast<const __half*>(o),
        static_cast<const __half*>(z), static_cast<const __half*>(w), static_cast<__half*>(y), rows, dv, eps);
}

} // namespace kernel
