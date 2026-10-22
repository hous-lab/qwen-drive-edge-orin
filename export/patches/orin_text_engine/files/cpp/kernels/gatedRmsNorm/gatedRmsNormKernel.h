/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace kernel
{

// y = RMSNormGated(o, z): o/z/w/y FP16; o/z [* rows, dv] contiguous, w [dv],
// y [* rows, dv]. Exact Qwen3_5RMSNormGated semantics (fp32 normalize ->
// round to input dtype -> * weight -> * silu(z fp32) -> round), eps = 1e-6.
void gatedRmsNormForward(void const* o, void const* z, void const* w, void* y, int64_t rows, int dv, float eps,
    cudaStream_t stream);

} // namespace kernel
