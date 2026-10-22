// pybind wrapper for the standalone CUDA DeltaNet chunk kernel (parity harness
// only -- the TRT plugin will call chunk_delta_cuda.h directly).
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <numeric>
#include <optional>

#include "chunk_delta_cuda_tco.h"

namespace {

constexpr int kChunk = 64;

std::vector<at::Tensor> run_prefill(const at::Tensor& q, const at::Tensor& k,
                                    const at::Tensor& v, const at::Tensor& g,
                                    const at::Tensor& beta, const at::Tensor* s0,
                                    std::vector<at::Tensor>* ws_out) {
    auto qc = q.contiguous();
    auto kc = k.contiguous();
    auto vc = v.contiguous();
    auto gc = g.contiguous();
    auto bc = beta.contiguous();

    const int B = qc.size(0), T = qc.size(1), HQK = qc.size(2), DK = qc.size(3);
    const int HV = vc.size(2), DV = vc.size(3);
    TORCH_CHECK(kc.size(2) == HQK && gc.size(2) == HV && bc.size(2) == HV,
                "head-count mismatch");
    TORCH_CHECK(HV % HQK == 0, "HV must be a multiple of HQK");

    const int NC = (T + kChunk - 1) / kChunk;
    const long bhnc = (long)B * HV * NC;
    const long ws_floats =
        bhnc * ((long)kChunk * DV + (long)kChunk * DK + 5L * kChunk * kChunk + 2L * kChunk + 1) +
        ((bhnc + 3) & ~3L) +  // alignment pad before Sc (must match launcher)
        (long)B * HV * NC * (long)DK * DV;  // Sc: per-chunk entry states (carry -> o pass)

    auto o = at::empty_like(vc);
    auto sf = at::empty({B, HV, DK, DV}, vc.options().dtype(at::kFloat));
    auto ws = at::empty({ws_floats}, vc.options().dtype(at::kFloat));

    const float* s0p = nullptr;
    at::Tensor s0c;
    if (s0) {
        s0c = s0->to(at::kFloat).contiguous();
        s0p = s0c.data_ptr<float>();
    }

    cudaError_t err = deltanet_tcop_chunk_prefill(
        qc.data_ptr(), kc.data_ptr(), vc.data_ptr(), gc.data_ptr<float>(), bc.data_ptr<float>(),
        o.data_ptr(), s0p, sf.data_ptr<float>(), B, T, HV, HQK, DK, DV, kChunk, ws.data_ptr(),
        (size_t)ws_floats * sizeof(float), at::cuda::getCurrentCUDAStream(),
        ws_out != nullptr ? 1 : 0);
    TORCH_CHECK(err == cudaSuccess, "deltanet_tcop_chunk_prefill failed: ", cudaGetErrorString(err));

    if (ws_out) {
        // carve the workspace into views:
        // [w | u | de | eg | ed | gl | Sc | M | T | A | T-post-leaf]  (order
        // must match the launcher's carve-out; M is the K3a tile dump, the
        // rest of T/A/T-post-leaf are debug)
        long off = 0;
        auto carve = [&](const std::vector<int64_t>& shape) {
            const int64_t n = std::accumulate(shape.begin(), shape.end(), int64_t(1),
                                              std::multiplies<int64_t>());
            auto t = ws.narrow(0, off, n);
            off += n;
            return t.reshape(shape);
        };
        ws_out->push_back(carve({B, HV, NC, kChunk, DV}));      // w
        ws_out->push_back(carve({B, HV, NC, kChunk, DK}));      // u
        ws_out->push_back(carve({B, HV, NC, kChunk, kChunk}));  // de
        ws_out->push_back(carve({B, HV, NC, kChunk}));          // eg
        ws_out->push_back(carve({B, HV, NC, kChunk}));          // ed
        ws_out->push_back(carve({B, HV, NC}));                  // gl
        {  // 16B alignment pad before Sc (mirrors the launcher carve)
            const long pad = ((long)B * HV * NC + 3) & ~3L;
            off += pad;
        }
        ws_out->push_back(carve({B, HV, NC, DK, DV}));          // Sc (entry states)
        ws_out->push_back(carve({B, HV, NC, kChunk, kChunk}));  // M (K3a tiles)
        ws_out->push_back(carve({B, HV, NC, kChunk, kChunk}));  // T (debug)
        ws_out->push_back(carve({B, HV, NC, kChunk, kChunk}));  // A (debug)
        ws_out->push_back(carve({B, HV, NC, kChunk, kChunk}));  // T post-leaf (debug)
    }
    return {o, sf};
}

std::vector<at::Tensor> chunk_prefill_impl(const at::Tensor& q, const at::Tensor& k,
                                           const at::Tensor& v, const at::Tensor& g,
                                           const at::Tensor& beta,
                                           const std::optional<at::Tensor>& s0,
                                           bool debug) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && g.is_cuda() && beta.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                    v.scalar_type() == at::kBFloat16,
                "q/k/v must be bf16");
    TORCH_CHECK(g.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat,
                "g/beta must be fp32");
    TORCH_CHECK(q.dim() == 4 && v.dim() == 4 && g.dim() == 3, "bad ranks");

    std::vector<at::Tensor> ws;
    auto out = run_prefill(q, k, v, g, beta, s0 ? &*s0 : nullptr, debug ? &ws : nullptr);
    if (debug) {
        out.insert(out.end(), ws.begin(), ws.end());
    }
    return out;
}

}  // namespace

std::vector<at::Tensor> chunk_prefill(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor g,
                                      at::Tensor beta, std::optional<at::Tensor> s0) {
    return chunk_prefill_impl(q, k, v, g, beta, s0, /*debug=*/false);
}

std::vector<at::Tensor> chunk_prefill_debug(at::Tensor q, at::Tensor k, at::Tensor v,
                                            at::Tensor g, at::Tensor beta,
                                            std::optional<at::Tensor> s0) {
    return chunk_prefill_impl(q, k, v, g, beta, s0, /*debug=*/true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace pybind11::literals;  // for ""_a
    m.def("chunk_prefill", &chunk_prefill,
          "chunked gated delta rule prefill (CUDA C++ port)", "q"_a, "k"_a, "v"_a, "g"_a,
          "beta"_a, "s0"_a = std::nullopt);
    m.def("chunk_prefill_debug", &chunk_prefill_debug,
          "same, also returns prep scratch (w, u, de, eg, ed, gl)", "q"_a, "k"_a, "v"_a, "g"_a,
          "beta"_a, "s0"_a = std::nullopt);
}
