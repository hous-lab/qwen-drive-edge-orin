// CUDA C++ port of the G0-passed Triton DeltaNet chunk kernel
// (../deltanet_tcop_kernel/chunk_delta.py). See chunk_delta_cuda.h for the IO
// contract. Kernel topology:
//   K1 "prep"  grid (NC, HV, B): all chunk-local math (A, (I-A)^-1 via 16x16
//              leaves + blockwise forward substitution, u, w) into fp32 scratch.
//   K2 "carry" grid (NSPLIT, HV, B): serial chunk walk per (b,h,DV-slice) that
//              advances only the recurrent state S and dumps every chunk's
//              entry state (fp32) so the per-chunk outputs decouple.
//   K3 "o"     grid (NSPLIT, HV, B*NC): fully parallel per-chunk output pass;
//              reads the dumped entry states and recomputes v_new/M/o for its
//              chunk. No serial dependency -> B*HV*NSPLIT*NC blocks instead of
//              16 waves of a 53-step serial loop.
//
// Numerics: everything fp32 FMA (superset of the shipped Triton tf32x3
// delivery config); the carry/o split is bitwise-identical to a fused serial
// scan (same dots, same operand order, fp32 round-trip through the dump).
// Tail chunk = zero-padded loads, which reproduces the fallback's F.pad
// semantics exactly (padded beta/g rows keep cumsum and decay constant and
// zero out their A/u/w contributions).
#include <cuda_bf16.h>
#include <math.h>

#include <mma.h>

#include "chunk_delta_cuda_tco.h"

namespace {

namespace wmma = nvcuda::wmma;

constexpr int NTHREADS = 256;
constexpr int BLK = 16;                       // inversion leaf size
// bf16 smem row stride for q/k staging: +2 elements makes the row word-offset
// odd, which kills the 32-way bank conflict of the q@k^T read pattern.
// k/q bf16 staging stride pad; 8 keeps the tile row pitch a multiple of 8
// elements, which the wmma load_matrix_sync ldm rule requires
constexpr int KSTRIDE_PAD = 8;

// C[M,N] = A[M,K] @ B[K,N]; operands/results in (dynamic) shared memory.
// Rows of C/A/B are addressed with ldc/lda/ldb respectively, so blocks carved
// out of a larger matrix (e.g. 16x16 leaves of the 64x64 Tinv) keep the parent
// stride. Syncs before/after: safe sequencing helper, C must not overlap A/B.
__device__ __forceinline__ void mm_smem_ex(float* C, int ldc, const float* A, int lda,
                                           const float* B, int ldb, int M, int N, int K) {
    __syncthreads();
    for (int e = threadIdx.x; e < M * N; e += NTHREADS) {
        int i = e / N;
        int j = e - i * N;
        float acc = 0.f;
        for (int d = 0; d < K; ++d) acc = fmaf(A[i * lda + d], B[d * ldb + j], acc);
        C[i * ldc + j] = acc;
    }
    __syncthreads();
}

// Dense-operand convenience wrapper (every row stride equals the matrix width).
__device__ __forceinline__ void mm_smem(float* C, const float* A, const float* B,
                                        int M, int N, int K) {
    mm_smem_ex(C, N, A, K, B, N, M, N, K);
}

// Same, but C lives in global memory with row stride ldc (used for u/w which
// are written straight out so they never need shared-memory staging). N must
// be a multiple of 4 and C 16B-aligned; each thread computes 4 consecutive
// columns and stores them as one float4.
__device__ __forceinline__ void mm_store_global(float* C, const float* A, const float* B,
                                                int M, int N, int K, int ldc) {
    __syncthreads();
    for (int e = threadIdx.x; e < M * (N / 4); e += NTHREADS) {
        const int i = e / (N / 4);
        const int j4 = (e - i * (N / 4)) * 4;
        const float* a = A + i * K;
        const float* b = B + j4;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll 8
        for (int d = 0; d < K; ++d) {
            const float av = a[d];
            a0 = fmaf(av, b[d * N + 0], a0);
            a1 = fmaf(av, b[d * N + 1], a1);
            a2 = fmaf(av, b[d * N + 2], a2);
            a3 = fmaf(av, b[d * N + 3], a3);
        }
        reinterpret_cast<float4*>(C + i * ldc + j4)[0] = make_float4(a0, a1, a2, a3);
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// K1: prep. One block per (chunk, head, batch).
// ---------------------------------------------------------------------------
template <int DK, int DV, int CS>
__global__ void __launch_bounds__(NTHREADS)
deltanet_tcop_prep_kernel(const __nv_bfloat16* __restrict__ kp,
                     const __nv_bfloat16* __restrict__ vp,
                     const float* __restrict__ gp,
                     const float* __restrict__ betap,
                     float* __restrict__ wp, float* __restrict__ up,
                     float* __restrict__ dep, float* __restrict__ egp,
                     float* __restrict__ edp, float* __restrict__ glp,
                     float* __restrict__ dbgtp,
                     float* __restrict__ dbgap, float* __restrict__ dbglp,
                     int T, int NC, int HV, int HQK) {
    const int cidx = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;
    const long qkh = h / (HV / HQK);  // repeat_interleave head mapping

    extern __shared__ char smem_raw[];
    float* c_s = reinterpret_cast<float*>(smem_raw);              // CS
    float* b_s = c_s + CS;                                        // CS
    __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(b_s + CS);  // CS*(DK+pad)
    float* Tinv = reinterpret_cast<float*>(ks + CS * (DK + KSTRIDE_PAD));  // CS*CS
    float* E = Tinv + CS * CS;                                    // CS*DK
    float* tmp1 = E + CS * DK;                                    // BLK*BLK
    float* tmp2 = tmp1 + BLK * BLK;                               // BLK*BLK

    // ---- g / beta (zero-padded tail) + in-chunk cumsum --------------------
    for (int i = tid; i < CS; i += NTHREADS) {
        const long tofs = (long)cidx * CS + i;
        const bool ok = tofs < T;
        c_s[i] = ok ? gp[((long)b * T + tofs) * HV + h] : 0.f;
        b_s[i] = ok ? betap[((long)b * T + tofs) * HV + h] : 0.f;
    }
    __syncthreads();
    if (tid == 0) {  // serial inclusive scan; 64 adds, off the scan kernel's path
        float acc = 0.f;
        for (int i = 0; i < CS; ++i) {
            acc += c_s[i];
            c_s[i] = acc;
        }
    }
    __syncthreads();
    const float g_last = c_s[CS - 1];

    // ---- stage k (bf16, padded rows -> 0); 16B-vector global loads --------
    for (int e = tid; e < CS * (DK / 8); e += NTHREADS) {
        const int i = e / (DK / 8);
        const int d8 = e - i * (DK / 8);
        const long tofs = (long)cidx * CS + i;
        union { uint4 u; __nv_bfloat16 h[8]; } pkt;
        if (tofs < T) {
            pkt.u = reinterpret_cast<const uint4*>(
                kp + (((long)b * T + tofs) * HQK + qkh) * DK)[d8];
        } else {
            pkt.u = make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int l = 0; l < 8; ++l) ks[i * (DK + KSTRIDE_PAD) + d8 * 8 + l] = pkt.h[l];
    }
    __syncthreads();

    // ---- A = -(k*beta) @ k^T * decay, strict lower, into Tinv buffer ------
    // (fallback L292; decay = exp(c_i - c_j) <= 1 since g <= 0)
    for (int e = tid; e < CS * CS; e += NTHREADS) {
        const int i = e / CS;
        const int j = e - i * CS;
        float val = 0.f;
        if (i > j) {
            const __nv_bfloat16* ki = ks + i * (DK + KSTRIDE_PAD);
            const __nv_bfloat16* kj = ks + j * (DK + KSTRIDE_PAD);
            float acc = 0.f;
#pragma unroll 8
            for (int d = 0; d < DK; ++d) {
                acc = fmaf(__bfloat162float(ki[d]) * b_s[i], __bfloat162float(kj[d]), acc);
            }
            val = -acc * expf(c_s[i] - c_s[j]);
        }
        Tinv[e] = val;
    }
    __syncthreads();

    if (dbgap != nullptr) {  // debug: dump assembled A
        float* dbga = dbgap + (((long)b * HV + h) * NC + cidx) * CS * CS;
        for (int e = tid; e < CS * CS; e += NTHREADS) dbga[e] = Tinv[e];
    }

    // ---- Tinv = (I - A)^{-1} ----------------------------------------------
    // Diagonal 16x16 leaves: T = sum_{i<16} A^i via 4 squaring steps
    // (identical to the Triton _leaf_inv; full-64 doubling is numerically
    // unsafe on correlated activations -- see chunk_delta.py notes).
    for (int bl = 0; bl < CS / BLK; ++bl) {
        float* Tslot = Tinv + bl * BLK * (CS + 1);  // rows/cols [bl*16,(bl+1)*16)
        // P = A_leaf: strided read of the parent-matrix block into dense tmp2
        for (int e = tid; e < BLK * BLK; e += NTHREADS) {
            const int i = e / BLK;
            const int j = e - i * BLK;
            tmp2[e] = Tslot[i * CS + j];
        }
        __syncthreads();
        for (int e = tid; e < BLK * BLK; e += NTHREADS) {
            const int i = e / BLK;
            const int j = e - i * BLK;
            Tslot[i * CS + j] = (i == j) ? 1.f : 0.f;  // T = I
        }
        __syncthreads();
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            mm_smem_ex(tmp1, BLK, Tslot, CS, tmp2, BLK, BLK, BLK, BLK);  // tmp1 = T @ P
            for (int e = tid; e < BLK * BLK; e += NTHREADS) {
                const int i = e / BLK;
                const int j = e - i * BLK;
                Tslot[i * CS + j] += tmp1[e];
            }
            __syncthreads();
            mm_smem(tmp1, tmp2, tmp2, BLK, BLK, BLK);  // tmp1 = P @ P (dense)
            for (int e = tid; e < BLK * BLK; e += NTHREADS) tmp2[e] = tmp1[e];
            __syncthreads();
        }
    }
    if (dbglp != nullptr) {  // debug: dump T right after the leaf inversion
        __syncthreads();
        float* dbgl = dbglp + (((long)b * HV + h) * NC + cidx) * CS * CS;
        for (int e = tid; e < CS * CS; e += NTHREADS) dbgl[e] = Tinv[e];
    }
    // Off-diagonal blocks: forward substitution, same formulas/order as the
    // Triton kernel (T10, T20, T21, T30, T31, T32), in place per block.
    {
#define SLOT(ii, jj) (Tinv + (ii)*BLK * CS + (jj)*BLK)
        // All block operands keep the parent row stride CS; tmp1/tmp2 are dense
        // BLKxBLK buffers (row stride BLK). C never overlaps A/B.
        // T10 = T11 @ (A10 @ T00)
        mm_smem_ex(tmp1, BLK, SLOT(1, 0), CS, SLOT(0, 0), CS, BLK, BLK, BLK);
        mm_smem_ex(SLOT(1, 0), CS, SLOT(1, 1), CS, tmp1, BLK, BLK, BLK, BLK);
        // T20 = T22 @ (A20 @ T00 + A21 @ T10)
        mm_smem_ex(tmp1, BLK, SLOT(2, 0), CS, SLOT(0, 0), CS, BLK, BLK, BLK);
        mm_smem_ex(tmp2, BLK, SLOT(2, 1), CS, SLOT(1, 0), CS, BLK, BLK, BLK);
        for (int e = tid; e < BLK * BLK; e += NTHREADS) tmp1[e] += tmp2[e];
        __syncthreads();
        mm_smem_ex(SLOT(2, 0), CS, SLOT(2, 2), CS, tmp1, BLK, BLK, BLK, BLK);
        // T21 = T22 @ (A21 @ T11)
        mm_smem_ex(tmp1, BLK, SLOT(2, 1), CS, SLOT(1, 1), CS, BLK, BLK, BLK);
        mm_smem_ex(SLOT(2, 1), CS, SLOT(2, 2), CS, tmp1, BLK, BLK, BLK, BLK);
        // T30 = T33 @ (A30 @ T00 + A31 @ T10 + A32 @ T20)
        mm_smem_ex(tmp1, BLK, SLOT(3, 0), CS, SLOT(0, 0), CS, BLK, BLK, BLK);
        mm_smem_ex(tmp2, BLK, SLOT(3, 1), CS, SLOT(1, 0), CS, BLK, BLK, BLK);
        for (int e = tid; e < BLK * BLK; e += NTHREADS) tmp1[e] += tmp2[e];
        __syncthreads();
        mm_smem_ex(tmp2, BLK, SLOT(3, 2), CS, SLOT(2, 0), CS, BLK, BLK, BLK);
        for (int e = tid; e < BLK * BLK; e += NTHREADS) tmp1[e] += tmp2[e];
        __syncthreads();
        mm_smem_ex(SLOT(3, 0), CS, SLOT(3, 3), CS, tmp1, BLK, BLK, BLK, BLK);
        // T31 = T33 @ (A31 @ T11 + A32 @ T21)
        mm_smem_ex(tmp1, BLK, SLOT(3, 1), CS, SLOT(1, 1), CS, BLK, BLK, BLK);
        mm_smem_ex(tmp2, BLK, SLOT(3, 2), CS, SLOT(2, 1), CS, BLK, BLK, BLK);
        for (int e = tid; e < BLK * BLK; e += NTHREADS) tmp1[e] += tmp2[e];
        __syncthreads();
        mm_smem_ex(SLOT(3, 1), CS, SLOT(3, 3), CS, tmp1, BLK, BLK, BLK, BLK);
        // T32 = T33 @ (A32 @ T22)
        mm_smem_ex(tmp1, BLK, SLOT(3, 2), CS, SLOT(2, 2), CS, BLK, BLK, BLK);
        mm_smem_ex(SLOT(3, 2), CS, SLOT(3, 3), CS, tmp1, BLK, BLK, BLK, BLK);
#undef SLOT
    }

    // ---- u = T @ (k*beta*exp(c))  (fallback k_cumdecay, L299) -------------
    for (int e = tid; e < CS * DK; e += NTHREADS) {
        const int i = e / DK;
        const int d = e - i * DK;
        E[e] = __bfloat162float(ks[i * (DK + KSTRIDE_PAD) + d]) * (b_s[i] * expf(c_s[i]));
    }
    __syncthreads();
    mm_store_global(up + (((long)b * HV + h) * NC + cidx) * CS * DK,
                    Tinv, E, CS, DK, CS, DK);

    // ---- w = T @ (v*beta)  (fallback "value", L298); vector v loads --------
    for (int e = tid; e < CS * (DV / 8); e += NTHREADS) {
        const int i = e / (DV / 8);
        const int d8 = e - i * (DV / 8);
        const long tofs = (long)cidx * CS + i;
        const float bi = b_s[i];
        union { uint4 u; __nv_bfloat16 h[8]; } pkt;
        pkt.u = (tofs < T) ? reinterpret_cast<const uint4*>(
                                 vp + (((long)b * T + tofs) * HV + h) * DV)[d8]
                           : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
        for (int l = 0; l < 8; ++l) E[i * DV + d8 * 8 + l] = __bfloat162float(pkt.h[l]) * bi;
    }
    __syncthreads();
    mm_store_global(wp + (((long)b * HV + h) * NC + cidx) * CS * DV,
                    Tinv, E, CS, DV, CS, DV);

    // ---- exp terms + decay matrix for the scan -----------------------------
    float* egc = egp + (((long)b * HV + h) * NC + cidx) * CS;
    float* edc = edp + (((long)b * HV + h) * NC + cidx) * CS;
    for (int i = tid; i < CS; i += NTHREADS) {
        egc[i] = expf(c_s[i]);
        edc[i] = expf(g_last - c_s[i]);
    }
    if (tid == 0) glp[((long)b * HV + h) * NC + cidx] = expf(g_last);
    float* dec = dep + (((long)b * HV + h) * NC + cidx) * CS * CS;
    for (int e = tid; e < CS * CS; e += NTHREADS) {
        const int i = e / CS;
        const int j = e - i * CS;
        dec[e] = (i >= j) ? expf(c_s[i] - c_s[j]) : 0.f;  // lower-inclusive, diag 1
    }

    // ---- debug: dump T = (I-A)^{-1} into the workspace tail ----------------
    if (dbgtp != nullptr) {
        __syncthreads();
        float* dbgc = dbgtp + (((long)b * HV + h) * NC + cidx) * CS * CS;
        for (int e = tid; e < CS * CS; e += NTHREADS) dbgc[e] = Tinv[e];
    }
}

// ---------------------------------------------------------------------------
// K2: carry. One block per (DV slice, head, batch); serial over chunks.
// Advances only the recurrent state, S = gl*S + (k*ed)^T @ (w - u @ S), and
// dumps every chunk's ENTRY state (the exact fp32 S a fused serial scan would
// consume) so the per-chunk outputs become independent work items for K3.
// All dots keep the fused scan's operand order -> bitwise-identical results.
// ---------------------------------------------------------------------------
template <int DK, int DV, int CS, int BV>
__global__ void __launch_bounds__(NTHREADS)
deltanet_tcop_carry_kernel(const __nv_bfloat16* __restrict__ kp,
                      const float* __restrict__ wp, const float* __restrict__ up,
                      const float* __restrict__ edp, const float* __restrict__ glp,
                      float* __restrict__ scp,
                      const float* __restrict__ s0p, float* __restrict__ sfp,
                      int T, int NC, int HV, int HQK) {
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;
    const int j0 = blockIdx.x * BV;
    const long qkh = h / (HV / HQK);

    extern __shared__ char smem_raw[];
    float* S = reinterpret_cast<float*>(smem_raw);                // DK*BV
    float* W = S + DK * BV;                                       // CS*BV
    float* u_s = W + CS * BV;                                     // CS*DK
    float* ed_s = u_s + CS * DK;                                  // CS
    __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(ed_s + CS);  // CS*(DK+pad)
    float4* S4 = reinterpret_cast<float4*>(S);
    float4* W4 = reinterpret_cast<float4*>(W);
    float4* u_s4 = reinterpret_cast<float4*>(u_s);

    // S init (s0 slice rows are DV-strided; vectorize within each row)
    for (int e = tid; e < DK * (BV / 4); e += NTHREADS) {
        const int d = e / (BV / 4);
        const int c4 = e - d * (BV / 4);
        if (s0p) {
            S4[e] = reinterpret_cast<const float4*>(
                s0p + (((long)b * HV + h) * DK + d) * DV + j0)[c4];
        } else {
            S4[e] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }

    for (int c = 0; c < NC; ++c) {
        const long bhnc = ((long)b * HV + h) * NC + c;
        const float* wc = wp + bhnc * CS * DV;
        const float* uc = up + bhnc * CS * DK;
        const float* edc = edp + bhnc * CS;
        const float glv = glp[bhnc];

        // stage u (contiguous CS*DK floats) and the w DV-slice (float4)
        for (int e = tid; e < CS * DK / 4; e += NTHREADS) u_s4[e] = reinterpret_cast<const float4*>(uc)[e];
        for (int e = tid; e < CS * (BV / 4); e += NTHREADS) {
            const int r = e / (BV / 4);
            const int c4 = e - r * (BV / 4);
            W4[e] = reinterpret_cast<const float4*>(wc + r * DV + j0)[c4];
        }
        for (int i = tid; i < CS; i += NTHREADS) ed_s[i] = edc[i];
        __syncthreads();

        // v_new = w - u @ S   (fallback L312-313); all operands in smem
        for (int e = tid; e < CS * BV; e += NTHREADS) {
            const int i = e / BV;
            const int j = e - i * BV;
            const float* ur = u_s + i * DK;
            float acc = 0.f;
#pragma unroll 8
            for (int d = 0; d < DK; ++d) acc = fmaf(ur[d], S[d * BV + j], acc);
            W[e] -= acc;
        }
        __syncthreads();  // W == v_new; S still holds the chunk-entry state

        // dump the entry state for K3 (fp32 round-trip is bitwise transparent)
        float* Sc = scp + (((long)b * HV + h) * NC + c) * DK * DV + j0;
        for (int e = tid; e < DK * (BV / 4); e += NTHREADS) {
            const int d = e / (BV / 4);
            const int c4 = e - d * (BV / 4);
            reinterpret_cast<float4*>(Sc + d * DV)[c4] = S4[e];
        }

        for (int e = tid; e < CS * (DK / 8); e += NTHREADS) {
            const int i = e / (DK / 8);
            const int d8 = e - i * (DK / 8);
            const long tofs = (long)c * CS + i;
            union { uint4 u; __nv_bfloat16 h[8]; } pkt;
            pkt.u = (tofs < T)
                        ? reinterpret_cast<const uint4*>(
                              kp + (((long)b * T + tofs) * HQK + qkh) * DK)[d8]
                        : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
            for (int l = 0; l < 8; ++l) ks[i * (DK + KSTRIDE_PAD) + d8 * 8 + l] = pkt.h[l];
        }
        __syncthreads();

        // S = S * gl + (k*ed)^T @ v_new   (fallback L316-319); float4 RMW
        for (int e = tid; e < DK * (BV / 4); e += NTHREADS) {
            const int d = e / (BV / 4);
            const int j4 = (e - d * (BV / 4)) * 4;
            float4 s = S4[e];
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll 8
            for (int i = 0; i < CS; ++i) {
                const float ked = __bfloat162float(ks[i * (DK + KSTRIDE_PAD) + d]) * ed_s[i];
                const float* wr = W + i * BV + j4;
                a0 = fmaf(ked, wr[0], a0);
                a1 = fmaf(ked, wr[1], a1);
                a2 = fmaf(ked, wr[2], a2);
                a3 = fmaf(ked, wr[3], a3);
            }
            S4[e] = make_float4(s.x * glv + a0, s.y * glv + a1, s.z * glv + a2, s.w * glv + a3);
        }
        __syncthreads();
    }

    for (int e = tid; e < DK * (BV / 4); e += NTHREADS) {
        const int d = e / (BV / 4);
        const int c4 = e - d * (BV / 4);
        reinterpret_cast<float4*>(sfp + (((long)b * HV + h) * DK + d) * DV + j0)[c4] = S4[e];
    }
}

// ---------------------------------------------------------------------------
// K3a: M-pass. One block per (chunk, head, batch): M = (q @ k^T) * dec * scale
// via bf16 wmma 16x16x16 (q/k are bf16 at the API boundary, so the HMMA
// products are exact; only the accumulation order differs from the scalar
// fp32 FMA ref), stored to the workspace so the o-pass's eight DV splits stop
// re-deriving it (that duplication was ~1/3 of the o-pass wall and kept the
// o-block smem above 64KB; offloading M drops the o-block to ~37KB = 5 per
// Orin SM). Same strict-lower decay mask, diagonal kept (fallback L311).
// ---------------------------------------------------------------------------
template <int DK, int CS>
__global__ void __launch_bounds__(NTHREADS)
deltanet_tcop_m_kernel(const __nv_bfloat16* __restrict__ qp,
                  const __nv_bfloat16* __restrict__ kp,
                  const float* __restrict__ dep,
                  float* __restrict__ mp,
                  int T, int NC, int HV, int HQK, float scale) {
    const int h = blockIdx.y;
    const int z = blockIdx.z;
    const int b = z / NC;
    const int c = z - b * NC;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const long qkh = h / (HV / HQK);
    const long bhnc = ((long)b * HV + h) * NC + c;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(smem_raw);   // CS*(DK+pad)
    __nv_bfloat16* ks = qs + CS * (DK + KSTRIDE_PAD);                 // CS*(DK+pad)
    float* M = reinterpret_cast<float*>(ks + CS * (DK + KSTRIDE_PAD));  // CS*CS
    float4* M4 = reinterpret_cast<float4*>(M);

    const int kpack = DK / 8;
    for (int e = tid; e < CS * kpack; e += NTHREADS) {
        const int i = e / kpack;
        const int d8 = e - i * kpack;
        const long tofs = (long)c * CS + i;
        const bool ok = tofs < T;
        union { uint4 u; __nv_bfloat16 h[8]; } pq, pk;
        pq.u = ok ? reinterpret_cast<const uint4*>(
                        qp + (((long)b * T + tofs) * HQK + qkh) * DK)[d8]
                  : make_uint4(0u, 0u, 0u, 0u);
        pk.u = ok ? reinterpret_cast<const uint4*>(
                        kp + (((long)b * T + tofs) * HQK + qkh) * DK)[d8]
                  : make_uint4(0u, 0u, 0u, 0u);
        __nv_bfloat16* qr = qs + i * (DK + KSTRIDE_PAD);
        __nv_bfloat16* kr = ks + i * (DK + KSTRIDE_PAD);
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            qr[d8 * 8 + l] = pq.h[l];
            kr[d8 * 8 + l] = pk.h[l];
        }
    }
    __syncthreads();

    // M_raw = q @ k^T (bf16 wmma, fp32 acc), then dec*scale post-pass with
    // the strict-lower decay mask; diagonal kept (fallback L311). M tiles are
    // 16x16 over a 64x64 result: 16 tiles, warp w handles tiles w and w+8.
    {
        const float* dec = dep + bhnc * CS * CS;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
#pragma unroll
        for (int t = warp; t < (CS / 16) * (CS / 16); t += NTHREADS / 32) {
            const int ti = t / (CS / 16);
            const int tj = t - ti * (CS / 16);
            wmma::fill_fragment(fc, 0.f);
#pragma unroll
            for (int d0 = 0; d0 < DK; d0 += 16) {
                wmma::load_matrix_sync(fa, qs + ti * 16 * (DK + KSTRIDE_PAD) + d0,
                                       DK + KSTRIDE_PAD);
                wmma::load_matrix_sync(fb, ks + tj * 16 * (DK + KSTRIDE_PAD) + d0,
                                       DK + KSTRIDE_PAD);
                wmma::mma_sync(fc, fa, fb, fc);
            }
            wmma::store_matrix_sync(M + ti * 16 * CS + tj * 16, fc, CS,
                                    wmma::mem_row_major);
        }
        __syncthreads();
        for (int e = tid; e < CS * CS; e += NTHREADS) {
            M[e] *= scale * dec[e];  // dec row-major CS x CS, matches layout
        }
        __syncthreads();
    }

    // store M to the workspace (float4; CS*CS is a multiple of 4 and the
    // carve keeps mp 16B-aligned)
    float* mc = mp + bhnc * CS * CS;
    for (int e = tid; e < (CS * CS) / 4; e += NTHREADS) {
        reinterpret_cast<float4*>(mc)[e] = M4[e];
    }
}

// ---------------------------------------------------------------------------
// K3b-TC: o-pass all-bf16 wmma (EXPERIMENT variant)
// K3b-TC: o-pass, all-bf16 wmma (EXPERIMENT variant of deltanet_tcop_o_kernel).
// Staging casts fp32 workspace operands (Sc/w/u/M) to bf16 in smem; dots on
// bf16 tensor cores (fp32 acc): v_new = u@S, term1 = q*e.g @ S, term2 = M@v_new.
// smem ~61KB -> same co-residency as canonical (2/SM x86, 3/SM Orin).
template <int DK, int DV, int CS, int BV>
__global__ void __launch_bounds__(NTHREADS)
deltanet_tcop_o_wmma_kernel(const __nv_bfloat16* __restrict__ qp,
                  const float* __restrict__ wp, const float* __restrict__ up,
                  const float* __restrict__ egp, const float* __restrict__ scp,
                  const float* __restrict__ mp,
                  __nv_bfloat16* __restrict__ op,
                  int T, int NC, int HV, int HQK, float scale, int null_op) {
    const int h = blockIdx.y;
    const int z = blockIdx.z;
    const int b = z / NC;
    const int c = z - b * NC;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int j0 = blockIdx.x * BV;
    const long qkh = h / (HV / HQK);
    const long bhnc = ((long)b * HV + h) * NC + c;
    extern __shared__ char smem_raw[];
    __nv_bfloat16* Ssb = reinterpret_cast<__nv_bfloat16*>(smem_raw);          // DK*BV
    __nv_bfloat16* Wb = Ssb + DK * BV;                                        // CS*BV
    float* Vnf = reinterpret_cast<float*>(Wb + CS * BV);                      // CS*BV
    __nv_bfloat16* Vnb = reinterpret_cast<__nv_bfloat16*>(Vnf + CS * BV);     // CS*BV
    __nv_bfloat16* ub = Vnb + CS * BV;                                        // CS*DK
    __nv_bfloat16* qs = ub + CS * DK;                                         // CS*(DK+pad)
    __nv_bfloat16* Mb = qs + CS * (DK + KSTRIDE_PAD);                         // CS*CS
    float* os1 = reinterpret_cast<float*>(Mb + CS * CS);                      // CS*BV
    float* os2 = os1 + CS * BV;                                               // CS*BV
    float* eg_s = os2 + CS * BV;                                              // CS
    for (int e = tid; e < DK * (BV / 2); e += NTHREADS) {
        const int d = e / (BV / 2);
        const int cv2 = (e - d * (BV / 2)) * 2;
        const float2 s2 = reinterpret_cast<const float2*>(scp + bhnc * DK * DV + d * DV + j0)[cv2 / 2];
        __nv_bfloat16* row = Ssb + d * BV;
        row[cv2] = __float2bfloat16(s2.x);
        row[cv2 + 1] = __float2bfloat16(s2.y);
    }
    for (int e = tid; e < CS * BV; e += NTHREADS) {
        const int i = e / BV;
        const int j = e - i * BV;
        Wb[e] = __float2bfloat16(wp[bhnc * CS * DV + i * DV + j0 + j]);
        eg_s[i] = egp[bhnc * CS + i] * scale;
    }
    for (int e = tid; e < CS * DK; e += NTHREADS) {
        const int i = e / DK;
        const int d = e - i * DK;
        ub[i * DK + d] = __float2bfloat16(up[bhnc * CS * DK + i * DK + d]);
    }
    const int kpack = DK / 8;
    for (int e = tid; e < CS * kpack; e += NTHREADS) {
        const int i = e / kpack;
        const int d8 = e - i * kpack;
        const long tofs = (long)c * CS + i;
        const bool ok = tofs < T;
        union { uint4 u; __nv_bfloat16 h[8]; } pq;
        pq.u = ok ? reinterpret_cast<const uint4*>(
                        qp + (((long)b * T + tofs) * HQK + qkh) * DK)[d8]
                  : make_uint4(0u, 0u, 0u, 0u);
        const float qeg = egp[bhnc * CS + i] * scale;
        __nv_bfloat16* qr = qs + i * (DK + KSTRIDE_PAD);
        for (int l = 0; l < 8; ++l) qr[d8 * 8 + l] = __float2bfloat16(__bfloat162float(pq.h[l]) * qeg);
    }
    for (int e = tid; e < (CS * CS) / 4; e += NTHREADS) {
        const int i = e / (CS / 4);
        const int j4 = (e - i * (CS / 4)) * 4;
        float4 m4 = reinterpret_cast<const float4*>(mp + bhnc * CS * CS + i * CS + j4)[0];
        __nv_bfloat16* row = Mb + i * CS + j4;
        row[0] = __float2bfloat16(m4.x);
        row[1] = __float2bfloat16(m4.y);
        row[2] = __float2bfloat16(m4.z);
        row[3] = __float2bfloat16(m4.w);
    }
    __syncthreads();
    if (!null_op) {
        // dot1: Vnf = ub @ Ssb (4 m-tiles, warps 0-3)
        if (warp < 4) {
            const int m0 = warp * 16;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> a1;
            wmma::fill_fragment(a1, 0.f);
            for (int k0 = 0; k0 < DK; k0 += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb;
                wmma::load_matrix_sync(fa, ub + m0 * DK + k0, DK);
                wmma::load_matrix_sync(fb, Ssb + k0 * BV, BV);
                wmma::mma_sync(a1, fa, fb, a1);
            }
            wmma::store_matrix_sync(Vnf + m0 * BV, a1, BV, wmma::mem_row_major);
        }
    }
    __syncthreads();
    for (int e = tid; e < CS * BV; e += NTHREADS) {
        const int i = e / BV;
        const int j = e - i * BV;
        Vnb[e] = __float2bfloat16(__bfloat162float(Wb[e]) - Vnf[e]);
    }
    if (!null_op) {
        // term1: os1 = qs @ Ssb ; term2: os2 = Mb @ Vnb (warps 0-3)
        if (warp < 4) {
            const int m0 = warp * 16;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> a2, a3;
            wmma::fill_fragment(a2, 0.f);
            wmma::fill_fragment(a3, 0.f);
            for (int k0 = 0; k0 < DK; k0 += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fq;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fs;
                wmma::load_matrix_sync(fq, qs + m0 * (DK + KSTRIDE_PAD) + k0, DK + KSTRIDE_PAD);
                wmma::load_matrix_sync(fs, Ssb + k0 * BV, BV);
                wmma::mma_sync(a2, fq, fs, a2);
            }
            for (int k0 = 0; k0 < CS; k0 += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fm;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fv;
                wmma::load_matrix_sync(fm, Mb + m0 * CS + k0, CS);
                wmma::load_matrix_sync(fv, Vnb + k0 * BV, BV);
                wmma::mma_sync(a3, fm, fv, a3);
            }
            wmma::store_matrix_sync(os1 + m0 * BV, a2, BV, wmma::mem_row_major);
            wmma::store_matrix_sync(os2 + m0 * BV, a3, BV, wmma::mem_row_major);
        }
    }
    __syncthreads();
    for (int e = tid; e < CS * BV; e += NTHREADS) {
        const int i = e / BV;
        const int j = e - i * BV;
        const long tofs = (long)c * CS + i;
        if (tofs < T) {
            op[(((long)b * T + tofs) * HV + h) * DV + j0 + j] = __float2bfloat16(os1[e] + os2[e]);
        }
    }
}


inline int cdiv(int a, int b) { return (a + b - 1) / b; }

}  // namespace

// ---------------------------------------------------------------------------
// Launcher. Workspace carve-out:
//   [ w: B*HV*NC*CS*DV | u: B*HV*NC*CS*DK | de: B*HV*NC*CS*CS
//     | eg: B*HV*NC*CS | ed: B*HV*NC*CS | gl: B*HV*NC
//     | Sc: B*HV*NC*DK*DV (chunk-entry states)
//     | M: B*HV*NC*CS*CS (K3a M tiles)
//     | T/A/T-post: 3 x B*HV*NC*CS*CS (debug dumps) ]  (all fp32)
// ---------------------------------------------------------------------------
size_t deltanet_tcop_prefill_workspace_size(int B, int T, int HV, int DK, int DV, int chunk_size) {
    const long NC = cdiv(T, chunk_size);
    const long bhnc = (long)B * HV * NC;
    const long floats =
        bhnc * ((long)chunk_size * DV + (long)chunk_size * DK + 5L * chunk_size * chunk_size +
                2L * chunk_size + 1) +
        ((long)B * HV * NC + 3 & ~3L) +  // alignment pad before Sc
        (long)B * HV * NC * (long)DK * DV;
    return (size_t)floats * sizeof(float);
}

cudaError_t deltanet_tcop_chunk_prefill(
    const void* q, const void* k, const void* v,
    const float* g, const float* beta,
    void* o, const float* initial_state, float* final_state,
    int B, int T, int HV, int HQK, int DK, int DV, int chunk_size,
    void* workspace, size_t workspace_bytes, cudaStream_t stream,
    int dump_debug) {
    if (DK != 128 || DV != 128 || chunk_size != 64) return cudaErrorInvalidValue;
    if (HQK <= 0 || HV % HQK != 0) return cudaErrorInvalidValue;
    if (workspace == nullptr) return cudaErrorInvalidValue;

    // The carry walk re-reads the whole CS*DK u block for its BV-slice every
    // chunk, but at BV_CARRY=64 the block hit ~117KB smem = one block per
    // Orin SM, leaving the 53-chunk serial walk latency-exposed (~245us per
    // step, ~100x its arithmetic). BV_CARRY=16 (~80KB -> 2 blocks/SM) buys
    // co-residency to hide that latency; the extra redundant u traffic is
    // far cheaper than the exposed stalls.
    constexpr int NSPLIT_CARRY = 8;
    constexpr int NSPLIT_O = 8;  // BV_O=16 floor: the tf32 term-2 wmma needs
                                 // 16-wide B tiles (BV<8 compiled with a div-
                                 // by-zero in the tile map and failed o cos
                                 // outright). With M offloaded to K3a the
                                 // o-block is ~54.5KB smem -> 3 blocks/SM
                                 // (pre-M-split it sat at 72KB -> 2/SM, and
                                 // the q-tile variant hit 100KB -> 1/SM,
                                 // serializing the o-pass to ~59ms alone).
    const int NC = cdiv(T, chunk_size);
    const size_t need = deltanet_tcop_prefill_workspace_size(B, T, HV, DK, DV, chunk_size);
    if (workspace_bytes < need) return cudaErrorInvalidValue;

    float* wp = (float*)workspace;
    float* up = wp + (long)B * HV * NC * chunk_size * DV;
    float* dep = up + (long)B * HV * NC * chunk_size * DK;
    float* egp = dep + (long)B * HV * NC * chunk_size * chunk_size;
    float* edp = egp + (long)B * HV * NC * chunk_size;
    float* glp = edp + (long)B * HV * NC * chunk_size;
    // Sc must stay 16B-aligned for its float4 access: B*HV*NC alone can be
    // odd (e.g. B=1, HV=2, NC=1), so round the gl->Sc seam up to 4 floats
    float* scp = glp + (((long)B * HV * NC + 3) & ~3L);
    float* mp = scp + (long)B * HV * NC * (long)DK * DV;  // K3a M tiles
    const long dbg_blk = (long)B * HV * NC * chunk_size * chunk_size;
    // debug dumps (T / A / post-leaf T) trail the live scratch; only written
    // when dump_debug is set, so the production path pays no dump bandwidth
    float* dbgtp = dump_debug ? mp + (long)B * HV * NC * chunk_size * chunk_size : nullptr;
    float* dbgap = dump_debug ? dbgtp + dbg_blk : nullptr;
    float* dbglp = dump_debug ? dbgap + dbg_blk : nullptr;

    constexpr int CS = 64;
    constexpr int BV_CARRY = 128 / NSPLIT_CARRY;  // carry DV slice
    constexpr int BV_O = 128 / NSPLIT_O;          // o-pass DV slice
    // shared-memory byte footprints (see kernel bodies for the layouts); fixed
    // to the validated DK=DV=128 config (enforced above). Carry stages the
    // CS*DK u block so its serial-path dots never touch global memory; the o
    // pass keeps u on the (parallel) global/L1 path and reads M from the
    // K3a workspace, keeping its footprint near 54 KB so three of its blocks
    // co-reside per SM.
    constexpr int PREP_SMEM =
        (2 * CS + CS * CS + CS * 128 + 2 * BLK * BLK) * 4 + CS * (128 + KSTRIDE_PAD) * 2;
    constexpr int CARRY_SMEM =
        (128 * BV_CARRY + CS * BV_CARRY + CS * 128 + CS) * 4 + CS * (128 + KSTRIDE_PAD) * 2;
    // S | W(v_new) | M | o_s | o_s2 | eg_s | qs   (M comes from K3a, not
    // recomputed per split; ks no longer staged)
    // K3b-TC layout: bf16(Ssb|Wb|Vnb|ub|qs|Mb) + f32(Vnf|os1|os2) + eg
    constexpr int O_SMEM = (128 * BV_O + 2 * CS * BV_O + CS * BV_O + CS * 128 +
                            CS * (128 + KSTRIDE_PAD) + CS * CS) * 2 +
                           3 * CS * BV_O * 4 + CS * 4;
    // qs | ks | M  (K3a M-pass block)
    constexpr int MS_SMEM = 2 * CS * (128 + KSTRIDE_PAD) * 2 + CS * CS * 4;  // 51200 B

    auto kern_prep = deltanet_tcop_prep_kernel<128, 128, 64>;
    auto kern_carry = deltanet_tcop_carry_kernel<128, 128, 64, BV_CARRY>;
    auto kern_m = deltanet_tcop_m_kernel<128, 64>;
    auto kern_o = deltanet_tcop_o_wmma_kernel<128, 128, 64, BV_O>;

    cudaError_t err = cudaFuncSetAttribute(kern_prep,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           PREP_SMEM);
    if (err != cudaSuccess) return err;
    err = cudaFuncSetAttribute(kern_carry,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               CARRY_SMEM);
    if (err != cudaSuccess) return err;
    err = cudaFuncSetAttribute(kern_m,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               MS_SMEM);
    if (err != cudaSuccess) return err;
    err = cudaFuncSetAttribute(kern_o,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               O_SMEM);
    if (err != cudaSuccess) return err;

    kern_prep<<<dim3(NC, HV, B), NTHREADS, PREP_SMEM, stream>>>(
        (const __nv_bfloat16*)k, (const __nv_bfloat16*)v, g, beta,
        wp, up, dep, egp, edp, glp, dbgtp, dbgap, dbglp, T, NC, HV, HQK);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    if (getenv("DTN_SKIP_CARRY") == nullptr) {
        kern_carry<<<dim3(NSPLIT_CARRY, HV, B), NTHREADS, CARRY_SMEM, stream>>>(
            (const __nv_bfloat16*)k, wp, up, edp, glp, scp, initial_state, final_state,
            T, NC, HV, HQK);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }

    // Per-kernel skip probes (timing-only; downstream operands go stale).
    const bool skip_o = getenv("DTN_SKIP_O") != nullptr;
    const bool skip_m = getenv("DTN_SKIP_M") != nullptr;
    if (!skip_m) {
        kern_m<<<dim3(1, HV, B * NC), NTHREADS, MS_SMEM, stream>>>(
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, dep, mp,
            T, NC, HV, HQK, 1.0f / sqrtf((float)DK));
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }
    if (!skip_o) {
        kern_o<<<dim3(NSPLIT_O, HV, B * NC), NTHREADS, O_SMEM, stream>>>(
            (const __nv_bfloat16*)q, wp, up, egp, scp, mp,
            (__nv_bfloat16*)o, T, NC, HV, HQK, 1.0f / sqrtf((float)DK),
            getenv("TCOP_NULL") != nullptr ? 1 : 0);
    }
    return cudaGetLastError();
}
