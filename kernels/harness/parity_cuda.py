"""Parity + latency: CUDA C++ DeltaNet port vs transformers torch fallback.

Standalone mode (no TRT, no Triton dependency in the kernel under test): the
kernel is compiled as a torch extension from chunk_delta_cuda.cu and compared
against the same reference used by deltanet_kernel/parity_test.py (installed
transformers fallback, or its verbatim copy baked into parity_test.py).

Run:
  # from kernels/ (compile is slow on Orin; TORCH_CUDA_ARCH_LIST=8.7)
  TORCH_CUDA_ARCH_LIST=8.7 python parity_cuda.py

Fixtures are the hard gate (same layer00/layer30 real activations that caught
the NaN on the first Triton version). Fixtures store RAW q/k (the call site
passes use_qk_l2norm_in_kernel=True), so this harness l2-normalizes q/k once in
fp32 and feeds the SAME tensors to reference and kernel -- the comparison then
isolates kernel math on the real-activation distribution (g cumsum to -1.7e4,
max|k_i.k_j| = 1.0 correlated rows).
"""
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
KERNEL_DIR = HERE.parent / "deltanet_kernel"
sys.path.insert(0, str(KERNEL_DIR))

from parity_test import (  # noqa: E402
    REF_SRC,
    _ref_chunk,
    bench,
    input_stats,
    load_fixture,
    make_inputs,
    max_rel,
    row_cos_min,
    row_cos_min_masked,
    tensor_l2_rel,
)


def build_extension():
    from torch.utils.cpp_extension import load

    if "orin" in torch.cuda.get_device_name(0).lower():
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.7")
    else:
        # x86 venv torch is cu124; force a matching nvcc (PATH may hold an
        # older toolkit first)
        os.environ["CUDA_HOME"] = "/usr/local/cuda-12.4"
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(HERE / ".ext_cache"))
    return load(
        name="deltanet_chunk_cuda",
        sources=[str(HERE / "chunk_delta_cuda.cu"), str(HERE / "torch_binding.cpp")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=True,
    )


def l2norm_fp32(x):
    inv = torch.rsqrt((x.float() * x.float()).sum(-1, keepdim=True) + 1e-6)
    return (x.float() * inv).to(torch.bfloat16)


def cuda_chunk(mod, q, k, v, g, beta, s0=None):
    """Kernel under test: bf16 in, (o bf16, final_state fp32) out."""
    o, sf = mod.chunk_prefill(q, k, v, g, beta, s0)
    return o, sf


def run_case(mod, B, S, l2_pre, with_init_state, do_time):
    q, k, v, g, beta, s0 = make_inputs(B, S, seed=hash((B, S, l2_pre)) % 2**31,
                                       l2_pre=l2_pre, with_init_state=with_init_state)
    qn, kn = (l2norm_fp32(q), l2norm_fp32(k)) if l2_pre else (q, k)
    # reference and kernel see IDENTICAL (already normalized) q/k: isolates math
    o_ref, st_ref = _ref_chunk(qn, kn, v, g=g, beta=beta, chunk_size=64,
                               initial_state=s0, output_final_state=True,
                               use_qk_l2norm_in_kernel=False)
    o_fast, st_fast = cuda_chunk(mod, qn, kn, v, g, beta, s0)
    res = {
        "o_cos": row_cos_min(o_fast, o_ref),
        "o_rel": max_rel(o_fast, o_ref),
        "st_cos": row_cos_min(st_fast, st_ref),
        "st_rel": max_rel(st_fast, st_ref),
    }
    nan = torch.isnan(o_fast.float()).any().item() or torch.isnan(st_fast).any().item()
    ok = (not nan) and res["o_cos"] >= 0.999 and res["o_rel"] < 1e-2 \
        and res["st_cos"] >= 0.999 and res["st_rel"] < 1e-2
    tag = f"S={S:5d} B={B} l2={'Y' if l2_pre else 'n'} init={'Y' if with_init_state else 'n'}"
    print(f"  [{tag}] o: cos={res['o_cos']:.6f} rel={res['o_rel']:.2e} | "
          f"state: cos={res['st_cos']:.6f} rel={res['st_rel']:.2e} | "
          f"{'PASS' if ok else 'FAIL' + (' (NaN!)' if nan else '')}", flush=True)
    ms = bench(lambda: cuda_chunk(mod, qn, kn, v, g, beta, s0)) if do_time else None
    return ok, ms


def run_fixture(mod, path, do_time=True):
    q, k, v, g, beta, kw = load_fixture(path)
    S = g.shape[1]
    qn, kn = l2norm_fp32(q), l2norm_fp32(k)
    s0 = kw.get("initial_state", None)
    o_ref, st_ref = _ref_chunk(qn, kn, v, g=g, beta=beta, output_final_state=True,
                               use_qk_l2norm_in_kernel=False)
    o_fast, st_fast = cuda_chunk(mod, qn, kn, v, g, beta, s0)
    nan_o = torch.isnan(o_fast.float()).any().item()
    nan_s = torch.isnan(st_fast).any().item()
    tag = f"{os.path.basename(path)} S={S}"
    ok = False
    if nan_o or nan_s:
        extra = f"NaN o={nan_o} state={nan_s}"
    else:
        o_cos, o_drop = row_cos_min_masked(o_fast, o_ref)
        o_l2 = tensor_l2_rel(o_fast, o_ref)
        st_cos = row_cos_min(st_fast, st_ref)
        st_rel = max_rel(st_fast, st_ref)
        # same hard gates as the Triton fixture check in parity_test.run_fixture
        ok = (o_l2 < 3e-3 and o_cos >= 0.9999 and st_cos >= 0.9999 and st_rel < 1e-3)
        extra = (f"o: l2={o_l2:.2e} cos={o_cos:.6f}(-{o_drop}) | "
                 f"state: cos={st_cos:.6f} rel={st_rel:.2e}")
    stats = input_stats(q, k, g, beta,
                        {"use_qk_l2norm_in_kernel": True})
    ms = None
    if do_time and ok:
        ms = bench(lambda: cuda_chunk(mod, qn, kn, v, g, beta, s0))
    print(f"  [{tag}] {extra} | {stats} | {'PASS' if ok else 'FAIL'}"
          + (f" cuda={ms:.2f}ms" if ms else ""), flush=True)
    return ok, ms


def main():
    assert torch.cuda.is_available(), "CUDA required"
    print(f"device={torch.cuda.get_device_name(0)}  torch={torch.__version__}")
    print(f"ref = torch fallback from {REF_SRC}")
    t0 = time.time()
    mod = build_extension()
    print(f"extension built in {time.time() - t0:.0f}s")

    all_ok = True
    rows = []
    fx_ms = {}
    print("\n== parity (random grid, chunk=64) ==")
    for l2_pre in (False, True):
        for B in (1, 2):
            for S in (128, 512, 3200):
                ok, ms = run_case(mod, B, S, l2_pre, with_init_state=False,
                                  do_time=(S == 3200))
                all_ok &= ok
                if ms is not None:
                    rows.append((S, B, "Y" if l2_pre else "n", ms))
    print("== extras ==")
    ok, _ = run_case(mod, 1, 512, False, with_init_state=True, do_time=False)
    all_ok &= ok

    print("\n== real-activation fixtures (captured from a demo-scene prefill) ==")
    for name in ("layer00.pt", "layer30.pt"):
        p = KERNEL_DIR / "fixtures" / name
        if not p.exists():
            print(f"  ({p} missing -- skip)")
            continue
        ok, ms = run_fixture(mod, p)
        all_ok &= ok
        if ms is not None:
            fx_ms[name] = ms

    print("\n== latency (single layer, chunk=64, CUDA C++) ==")
    print(f"{'S':>6} {'B':>2} {'l2':>2} | {'ms':>8}")
    for S, B, l2tag, ms in rows:
        print(f"{S:6d} {B:2d} {l2tag:>2} | {ms:8.3f}")
    for name, ms in fx_ms.items():
        print(f"{'fixture':>6} {1:2d} {'fx':>2} | {ms:8.3f}  ({name})")

    print(f"\nRESULT: {'ALL PASS' if all_ok else 'FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
