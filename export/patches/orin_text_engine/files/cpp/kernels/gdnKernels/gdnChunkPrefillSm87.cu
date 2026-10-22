/*
 * E3 Qwen-Drive text engine: sm_87 GDN prefill via the self-developed chunked
 * delta-rule kernel (chunk_delta_cuda.cu, bf16 wmma), bridged from the TRT
 * plugin's fp16 GDNParams I/O.
 *
 * Why: the official CuTe prefill kernel produces ALL-ZERO output on sm_87
 * (decode passes; see patches/README.md).  This adapter:
 *   1. L2-normalizes q/k (fp16 in, fp32 math, eps 1e-6) and converts to bf16.
 *   2. Converts v fp16 -> bf16.
 *   3. Computes g = -exp(A_log) * softplus(a + dt_bias)  (fp32, negative
 *      log-decay) and beta = sigmoid(b)  (fp32).
 *   4. Calls deltanet_chunk_prefill (bf16 IO, fp32 g/beta/s0/sf).
 *   5. Converts o bf16 -> fp16 and copies final state (fp32) into the
 *      working state buffer (params.h0_source, which the plugin has already
 *      seeded with the initial state).
 *
 * Scratch (bf16 q/k/v/o, g, beta, final state, kernel workspace) is one
 * growth-only cudaMalloc shared by all plugin instances (layers run
 * sequentially inside one TRT enqueue).  Exact-T contract: all batch items
 * have valid length == seq_len (the engine executes prefill with the true
 * token count; production batch = 1).
 */

#include "gdnChunkPrefillSm87.h"

#include "chunk_delta_cuda.h"
#include "common/logger.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <mutex>
#include <vector>

namespace trt_edgellm
{
namespace
{

constexpr int kDK = 128;
constexpr int kDV = 128;
constexpr int kChunk = 64;

__device__ __forceinline__ float softplusf(float x)
{
    // log(1 + e^x), overflow-safe; matches torch.nn.functional.softplus
    // (default beta=1, threshold=20) closely enough for fp32 gate math.
    return x > 20.f ? x : log1pf(expf(x));
}

// One row of 128 fp16 -> L2-normalize in fp32 -> one row of 128 bf16.
__global__ void l2normFp16ToBf16(void const* in, void* out, int totalRows)
{
    int row = blockIdx.x;
    if (row >= totalRows)
    {
        return;
    }
    __half const* src = static_cast<__half const*>(in) + static_cast<size_t>(row) * kDK;
    __nv_bfloat16* dst = static_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * kDK;

    __shared__ float sRed[32];
    float v = __half2float(src[threadIdx.x]);
    float sq = v * v;

    // block = 128 threads = 4 warps: warp reduce + cross-warp reduce
    for (int off = 16; off > 0; off >>= 1)
    {
        sq += __shfl_down_sync(0xffffffff, sq, off);
    }
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0)
    {
        sRed[warp] = sq;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float total = sRed[0] + sRed[1] + sRed[2] + sRed[3];
        sRed[0] = rsqrtf(total + 1e-6f);
    }
    __syncthreads();
    dst[threadIdx.x] = __float2bfloat16(v * sRed[0]);
}

// Elementwise fp16 -> bf16.
__global__ void convertFp16ToBf16(void const* in, void* out, long n)
{
    long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    static_cast<__nv_bfloat16*>(out)[i] =
        __float2bfloat16(__half2float(static_cast<__half const*>(in)[i]));
}

// Elementwise bf16 -> fp16.
__global__ void convertBf16ToFp16(void const* in, void* out, long n)
{
    long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    static_cast<__half*>(out)[i] =
        __float2half(__bfloat162float(static_cast<__nv_bfloat16 const*>(in)[i]));
}

// g[b,t,h]  = -exp(A_log[h]) * softplus(a[b,t,h] + dt_bias[h])   (fp32)
// beta[b,t,h] = sigmoid(b[b,t,h])                                 (fp32)
__global__ void computeGdnGate(__half const* a, __half const* b, float const* A_log,
    __half const* dt_bias, float* g, float* beta, long total, int hv)
{
    long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total)
    {
        return;
    }
    int h = static_cast<int>(i % hv);
    float av = __half2float(a[i]) + __half2float(dt_bias[h]);
    g[i] = -expf(A_log[h]) * softplusf(av);
    beta[i] = 1.f / (1.f + expf(-__half2float(b[i])));
}

struct ChunkPrefillScratch
{
    void* base{nullptr};
    size_t bytes{0};
    // carved pointers
    void *qBf, *kBf, *vBf, *oBf;
    float *g, *beta, *sf;
    void* ws;

    bool ensure(size_t n, size_t seqLen, size_t hv, size_t hk)
    {
        auto align256 = [](size_t x) { return (x + 255u) & ~size_t{255u}; };
        size_t qBytes = align256(n * seqLen * hk * kDK * 2);
        size_t kBytes = align256(n * seqLen * hk * kDK * 2);
        size_t vBytes = align256(n * seqLen * hv * kDV * 2);
        size_t oBytes = align256(n * seqLen * hv * kDV * 2);
        size_t gBytes = align256(n * seqLen * hv * 4);
        size_t betaBytes = align256(n * seqLen * hv * 4);
        size_t sfBytes = align256(n * hv * kDK * kDV * 4);
        size_t wsBytes = align256(
            deltanet_prefill_workspace_size(static_cast<int>(n), static_cast<int>(seqLen),
                static_cast<int>(hv), kDK, kDV, kChunk));
        size_t total = qBytes + kBytes + vBytes + oBytes + gBytes + betaBytes + sfBytes + wsBytes;
        if (total <= bytes)
        {
            return true;
        }
        if (base != nullptr)
        {
            // Growth is rare; make the free safe against still-pending work.
            cudaDeviceSynchronize();
            cudaFree(base);
        }
        cudaError_t err = cudaMalloc(&base, total);
        if (err != cudaSuccess)
        {
            base = nullptr;
            bytes = 0;
            return false;
        }
        bytes = total;
        char* p = static_cast<char*>(base);
        qBf = p; p += qBytes;
        kBf = p; p += kBytes;
        vBf = p; p += vBytes;
        oBf = p; p += oBytes;
        g = reinterpret_cast<float*>(p); p += gBytes;
        beta = reinterpret_cast<float*>(p); p += betaBytes;
        sf = reinterpret_cast<float*>(p); p += sfBytes;
        ws = p;
        return true;
    }
};

std::mutex gChunkMutex;
ChunkPrefillScratch gScratch;

} // namespace

int runGdnChunkPrefillSm87(GDNParams const& params, cudaStream_t stream)
{
    if (params.k_dim != kDK || params.v_dim != kDV)
    {
        LOG_ERROR("gdnChunkPrefillSm87: requires k_dim=v_dim=128, got %d/%d", params.k_dim, params.v_dim);
        return -1;
    }
    if (params.h <= 0 || params.hv <= 0 || params.hv < params.h || (params.hv % params.h) != 0)
    {
        LOG_ERROR("gdnChunkPrefillSm87: requires hv >= h and hv %% h == 0, got h=%d hv=%d", params.h, params.hv);
        return -1;
    }
    if (params.q == nullptr || params.k == nullptr || params.v == nullptr || params.a == nullptr
        || params.b == nullptr || params.A_log == nullptr || params.dt_bias == nullptr
        || params.h0_source == nullptr || params.o == nullptr)
    {
        LOG_ERROR("gdnChunkPrefillSm87: null GDNParams tensor pointer.");
        return -1;
    }

    int32_t const n = params.n;
    int32_t const seqLen = params.seq_len;
    int32_t const hk = params.h;
    int32_t const hv = params.hv;
    if (n <= 0 || seqLen <= 0)
    {
        LOG_ERROR("gdnChunkPrefillSm87: n and seq_len must be positive, got n=%d seq_len=%d", n, seqLen);
        return -1;
    }

    size_t const gateTotal = static_cast<size_t>(n) * seqLen * hv;
    long const qkRows = static_cast<long>(n) * seqLen * hk;
    long const vElems = static_cast<long>(n) * seqLen * hv * kDV;
    size_t const stateElems = static_cast<size_t>(n) * hv * kDK * kDV;

    {
        std::lock_guard<std::mutex> lock(gChunkMutex);
        if (!gScratch.ensure(n, seqLen, hv, hk))
        {
            LOG_ERROR("gdnChunkPrefillSm87: scratch cudaMalloc failed (%.1f MB wanted)",
                static_cast<double>(n * seqLen * (hk + hv) * kDK * 2 + n * seqLen * hv * 8
                        + n * hv * kDK * kDV * 4)
                    / 1048576.0);
            return -1;
        }

        auto launchEw = [&](void const* in, void* out, long elems) {
            int block = 256;
            long grid = (elems + block - 1) / block;
            if (grid > 0x7fffffffLL)
            {
                LOG_ERROR("gdnChunkPrefillSm87: grid too large (%ld)", grid);
                return false;
            }
            convertFp16ToBf16<<<static_cast<unsigned int>(grid), block, 0, stream>>>(in, out, elems);
            return cudaGetLastError() == cudaSuccess;
        };

        // q/k: fp16 -> (fp32 L2 norm, eps 1e-6) -> bf16.  Our kernel expects
        // pre-normalized q/k (use_qk_l2norm_in_kernel semantics).
        l2normFp16ToBf16<<<static_cast<unsigned int>(qkRows), 128, 0, stream>>>(
            params.q, gScratch.qBf, static_cast<int>(qkRows));
        if (cudaGetLastError() != cudaSuccess)
        {
            LOG_ERROR("gdnChunkPrefillSm87: q l2norm launch failed");
            return -1;
        }
        l2normFp16ToBf16<<<static_cast<unsigned int>(qkRows), 128, 0, stream>>>(
            params.k, gScratch.kBf, static_cast<int>(qkRows));
        if (cudaGetLastError() != cudaSuccess)
        {
            LOG_ERROR("gdnChunkPrefillSm87: k l2norm launch failed");
            return -1;
        }

        // v: plain fp16 -> bf16.
        if (!launchEw(params.v, gScratch.vBf, vElems))
        {
            LOG_ERROR("gdnChunkPrefillSm87: v convert failed");
            return -1;
        }

        // gate math (fp32).
        {
            int block = 256;
            long grid = (static_cast<long>(gateTotal) + block - 1) / block;
            computeGdnGate<<<static_cast<unsigned int>(grid), block, 0, stream>>>(
                static_cast<__half const*>(params.a), static_cast<__half const*>(params.b),
                static_cast<float const*>(params.A_log), static_cast<__half const*>(params.dt_bias),
                gScratch.g, gScratch.beta, static_cast<long>(gateTotal), hv);
            if (cudaGetLastError() != cudaSuccess)
            {
                LOG_ERROR("gdnChunkPrefillSm87: gate kernel launch failed");
                return -1;
            }
        }

        // Chunked delta rule.  h0_source already holds the INITIAL state
        // (plugin D2D-copied it into h0_out); final state goes to scratch and
        // is copied back stream-ordered to avoid any s0/sf aliasing assumption.
        cudaError_t kerr = deltanet_chunk_prefill(gScratch.qBf, gScratch.kBf, gScratch.vBf, gScratch.g,
            gScratch.beta, gScratch.oBf, static_cast<float const*>(params.h0_source), gScratch.sf, n, seqLen,
            hv, hk, kDK, kDV, kChunk, gScratch.ws,
            deltanet_prefill_workspace_size(n, seqLen, hv, kDK, kDV, kChunk), stream, /*dump_debug=*/0);
        if (kerr != cudaSuccess)
        {
            LOG_ERROR("gdnChunkPrefillSm87: deltanet_chunk_prefill failed: %s", cudaGetErrorString(kerr));
            return -1;
        }

        // o: bf16 scratch -> fp16 output (opposite direction of the v convert).
        {
            int block = 256;
            long grid = (vElems + block - 1) / block;
            convertBf16ToFp16<<<static_cast<unsigned int>(grid), block, 0, stream>>>(
                gScratch.oBf, params.o, vElems);
            if (cudaGetLastError() != cudaSuccess)
            {
                LOG_ERROR("gdnChunkPrefillSm87: o convert failed");
                return -1;
            }
        }

        // final state -> working state buffer.
        cudaError_t cerr = cudaMemcpyAsync(
            params.h0_source, gScratch.sf, stateElems * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        if (cerr != cudaSuccess)
        {
            LOG_ERROR("gdnChunkPrefillSm87: state copy-back failed: %s", cudaGetErrorString(cerr));
            return -1;
        }
    }
    return 0;
}

} // namespace trt_edgellm
