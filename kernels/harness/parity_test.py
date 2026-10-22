"""Parity + latency: custom Triton DeltaNet kernel vs transformers torch fallback.

Run (standalone, no package import needed):
  python harness/parity_test.py   # Triton reference kernel vs torch fallback

Pass criteria per case (o and final_state): min per-row cosine >= 0.999 and
max relative err < 1e-2 (denominator floored at 0.1 so near-zero elements are
judged in absolute terms). Prints a latency table at the end.
"""
import glob
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chunk_delta import chunk_gated_delta_rule_fast  # noqa: E402

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule as _ref_chunk,
        l2norm as _ref_l2norm,
    )
    REF_SRC = "transformers.models.qwen3_5 (installed)"
except Exception:  # pragma: no cover - fallback for machines without transformers
    REF_SRC = "local verbatim copy of modeling_qwen3_5.py"

    def _ref_l2norm(x, dim=-1, eps=1e-6):
        inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
        return x * inv_norm

    def _ref_chunk(query, key, value, g, beta, chunk_size=64, initial_state=None,
                   output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        # Verbatim math from transformers 5.14.1 modeling_qwen3_5.py L248-326.
        initial_dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            query = _ref_l2norm(query, dim=-1, eps=1e-6)
            key = _ref_l2norm(key, dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
        ]
        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
        total_sequence_length = sequence_length + pad_size
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale
        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        query, key, value, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
        last_recurrent_state = (
            torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
            if initial_state is None
            else initial_state.to(value)
        )
        core_attn_out = torch.zeros_like(value)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
        for i in range(0, total_sequence_length // chunk_size):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
            v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            core_attn_out[:, :, i] = attn_inter + attn @ v_new
            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
            )
        if not output_final_state:
            last_recurrent_state = None
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state


H, DK, DV = 32, 128, 128


def make_inputs(B, S, seed, l2_pre=False, with_init_state=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    # NOTE: raw mode uses N(0, 1/8) magnitudes. Unscaled N(0,1) q/k without
    # l2norm makes (I-A)^{-1} pathological -- even the official fp32 fallback
    # produces NaN there (max|A| ~ 45, Tinv explodes). The real call site
    # always runs with use_qk_l2norm_in_kernel=True, so this bounded regime
    # is the meaningful "random input" coverage.
    q = torch.randn(B, S, H, DK, device="cuda", dtype=torch.bfloat16, generator=gen) / 8.0
    k = torch.randn(B, S, H, DK, device="cuda", dtype=torch.bfloat16, generator=gen) / 8.0
    v = torch.randn(B, S, H, DV, device="cuda", dtype=torch.bfloat16, generator=gen)
    # realistic magnitudes: g = negative log-decay (<=0), beta in (0,1) like sigmoid
    g = -(torch.rand(B, S, H, device="cuda", generator=gen) * 0.3)
    beta = torch.rand(B, S, H, device="cuda", generator=gen)
    if l2_pre:
        q = _ref_l2norm(q.float()).to(torch.bfloat16)
        k = _ref_l2norm(k.float()).to(torch.bfloat16)
    s0 = None
    if with_init_state:
        s0 = torch.randn(B, H, DK, DV, device="cuda", dtype=torch.float32, generator=gen) * 0.5
    return q, k, v, g, beta, s0


def row_cos_min(a, b):
    a = a.float().reshape(-1, a.shape[-1])
    b = b.float().reshape(-1, b.shape[-1])
    return F.cosine_similarity(a, b, dim=-1).min().item()


def max_rel(a, b, floor=0.1):
    d = (a.float() - b.float()).abs()
    return (d / b.float().abs().clamp_min(floor)).max().item()


def tensor_l2_rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def row_cos_min_masked(a, b, floor_ratio=1e-3):
    """Min cosine over rows whose ref norm is not numerically degenerate.

    Real activations include rows whose magnitude underflows (g cumsum reaches
    -1.6e4 -> exp ~ 0); cosine on those rows is pure noise on both sides.
    Returns (min_cos, n_dropped_rows).
    """
    a = a.float().reshape(-1, a.shape[-1])
    b = b.float().reshape(-1, b.shape[-1])
    rn = b.norm(dim=-1)
    mask = rn > rn.max() * floor_ratio
    n_drop = int((~mask).sum())
    if not mask.any():
        return 1.0, n_drop
    return F.cosine_similarity(a[mask], b[mask], dim=-1).min().item(), n_drop


def bench(fn, est=None):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    if est is None:
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        est = time.perf_counter() - t0
    iters = max(3, min(20, int(1.5 / max(est, 1e-4))))
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def run_case(B, S, l2_pre, with_init_state, use_flag, do_time):
    q, k, v, g, beta, s0 = make_inputs(B, S, seed=hash((B, S, l2_pre)) % 2**31,
                                       l2_pre=l2_pre, with_init_state=with_init_state)
    o_ref, st_ref = _ref_chunk(q, k, v, g=g, beta=beta, chunk_size=64, initial_state=s0,
                               output_final_state=True, use_qk_l2norm_in_kernel=use_flag)
    o_fast, st_fast = chunk_gated_delta_rule_fast(q, k, v, g=g, beta=beta, chunk_size=64,
                                                  initial_state=s0, output_final_state=True,
                                                  use_qk_l2norm_in_kernel=use_flag)
    res = {
        "o_cos": row_cos_min(o_fast, o_ref),
        "o_rel": max_rel(o_fast, o_ref),
        "st_cos": row_cos_min(st_fast, st_ref),
        "st_rel": max_rel(st_fast, st_ref),
    }
    nan = torch.isnan(o_fast).any().item() or torch.isnan(st_fast).any().item()
    ok = (not nan) and res["o_cos"] >= 0.999 and res["o_rel"] < 1e-2 \
        and res["st_cos"] >= 0.999 and res["st_rel"] < 1e-2
    tag = f"S={S:5d} B={B} l2={'Y' if (l2_pre or use_flag) else 'n'} init={'Y' if with_init_state else 'n'}"
    print(f"  [{tag}] o: cos={res['o_cos']:.6f} rel={res['o_rel']:.2e} | "
          f"state: cos={res['st_cos']:.6f} rel={res['st_rel']:.2e} | "
          f"{'PASS' if ok else 'FAIL' + (' (NaN!)' if nan else '')}")
    ms_fast = ms_ref = None
    if do_time and ok is not None:
        ms_fast = bench(lambda: chunk_gated_delta_rule_fast(q, k, v, g=g, beta=beta, chunk_size=64,
                                                            initial_state=s0, output_final_state=True))
        ms_ref = bench(lambda: _ref_chunk(q, k, v, g=g, beta=beta, chunk_size=64, initial_state=s0,
                                          output_final_state=True, use_qk_l2norm_in_kernel=use_flag))
    return ok, ms_fast, ms_ref


def load_fixture(path):
    fx = torch.load(path, map_location="cpu", weights_only=False)
    q, k, v = [t.cuda() for t in fx["args"]]
    kw = {kk: (vv.cuda() if torch.is_tensor(vv) else vv) for kk, vv in fx["kwargs"].items()}
    g = kw.pop("g")
    beta = kw.pop("beta")
    kw.pop("output_final_state", None)  # passed explicitly at the call sites
    return q, k, v, g, beta, kw


def input_stats(q, k, g, beta, kw):
    """Distribution facts that bear on (I-A)^{-1} conditioning."""
    with torch.no_grad():
        gc = g.float()
        gcum = gc.cumsum(dim=1)
        bf = beta.float()
        kf = k.float()
        if kw.get("use_qk_l2norm_in_kernel", False):
            kf = _ref_l2norm(kf)
        h0 = kf[:, :, 0]  # [B,T,DK] head 0 as a representative
        T = h0.shape[1]
        sim = (h0 @ h0.transpose(-1, -2)).abs()
        tri = torch.tril(torch.ones(T, T, dtype=torch.bool, device=sim.device), diagonal=-1)
        return (f"g[{gc.min().item():.3f},{gc.max().item():.3f}] gcum_min={gcum.min().item():.1f} "
                f"beta[{bf.min().item():.3f},{bf.max().item():.3f}] "
                f"max|k_i.k_j|={sim.masked_select(tri).max().item():.4f}")


def localize_nan_prefix(q, k, v, g, beta, kw, S):
    """Binary-search the minimal prefix length whose run NaNs -> pins the first
    bad chunk ((L-1)//64). State depends only on the prefix, so if NO prefix
    NaNs, the NaN emerges inside the final chunk from accumulated drift."""
    def bad(L):
        o, st = chunk_gated_delta_rule_fast(q[:, :L], k[:, :L], v[:, :L], g=g[:, :L],
                                            beta=beta[:, :L], chunk_size=64,
                                            output_final_state=True, **kw)
        return bool(torch.isnan(o).any() or torch.isnan(st).any())
    lo, hi = 64, S
    if not bad(hi):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if bad(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def run_fixture(path, do_time=True):
    q, k, v, g, beta, kw = load_fixture(path)
    S = g.shape[1]
    o_ref, st_ref = _ref_chunk(q, k, v, g=g, beta=beta, output_final_state=True, **kw)
    o_fast, st_fast = chunk_gated_delta_rule_fast(q, k, v, g=g, beta=beta,
                                                  output_final_state=True, **kw)
    nan_o = torch.isnan(o_fast.float()).any().item()
    nan_s = torch.isnan(st_fast).any().item()
    tag = f"{os.path.basename(path)} S={S}"
    stats = input_stats(q, k, g, beta, kw)
    ok = False
    extra = ""
    if nan_o or nan_s:
        first = localize_nan_prefix(q, k, v, g, beta, kw, S)
        extra = f" | NaN o={nan_o} state={nan_s} first-bad-prefix={first}" \
                f" (chunk {(first - 1) // 64 if first else '?'}/{(S + 63) // 64})"
    else:
        o_cos, o_drop = row_cos_min_masked(o_fast, o_ref)
        o_l2 = tensor_l2_rel(o_fast, o_ref)
        st_cos, st_rel = row_cos_min(st_fast, st_ref), max_rel(st_fast, st_ref)
        # o: bf16 on BOTH sides -> per-element diff >= 1 ulp (~3.9e-3 rel), and
        # some rows are degenerate (see row_cos_min_masked), so gate on tensor
        # L2 rel + cos over non-degenerate rows. The shipped config keeps the
        # scan's three terminal o dots in tf32 (x3 there costs scan 5.8->80ms
        # on Orin for a bf16-invisible gain), which pins o l2 at ~2e-3 = half
        # a bf16 ulp; the model-level G0 regression validated that choice
        # (KV cos >= 0.9996, traj cos 0.99997). state: fp32, strict gates --
        # rel<=1e-3 is the 24-layer drift budget (shipped config: ~1e-5).
        ok = (o_l2 < 3e-3 and o_cos >= 0.9999 and st_cos >= 0.9999 and st_rel < 1e-3)
        extra = (f"o: l2={o_l2:.2e} cos={o_cos:.6f}(-{o_drop}) | "
                 f"state: cos={st_cos:.6f} rel={st_rel:.2e}")
    mf = mr = None
    if do_time and ok:
        mf = bench(lambda: chunk_gated_delta_rule_fast(q, k, v, g=g, beta=beta,
                                                       output_final_state=True, **kw))
        mr = bench(lambda: _ref_chunk(q, k, v, g=g, beta=beta, output_final_state=True, **kw))
    print(f"  [{tag}] {extra} | {stats} | "
          f"{'PASS' if ok else 'FAIL'}"
          + (f" fast={mf:.2f}ms ref={mr:.2f}ms" if mf else ""), flush=True)
    return ok, mf, mr


def main():
    assert torch.cuda.is_available(), "CUDA required"
    import triton
    print(f"device={torch.cuda.get_device_name(0)}  torch={torch.__version__}  triton={triton.__version__}")
    print(f"ref = torch_chunk_gated_delta_rule from {REF_SRC}")
    print(f"heads H={H} (q/k pre-expanded), DK=DV={DK}, chunk=64")

    all_ok = True
    rows = []
    print("\n== parity (random + l2norm'd inputs, S x B grid) ==")
    for l2_pre in (False, True):
        for B in (1, 2):
            for S in (128, 512, 3200):
                ok, mf, mr = run_case(B, S, l2_pre, with_init_state=False, use_flag=False,
                                      do_time=(S == 3200))
                all_ok &= ok
                if mf is not None:
                    rows.append((S, B, "Y" if l2_pre else "n", mf, mr))

    print("\n== extras ==")
    ok, _, _ = run_case(1, 512, False, with_init_state=True, use_flag=False, do_time=False)
    all_ok &= ok
    ok, _, _ = run_case(1, 512, False, with_init_state=False, use_flag=True, do_time=False)
    all_ok &= ok

    print("\n== real-activation fixtures (captured from a demo-scene prefill) ==")
    fix_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    fix_paths = sorted(glob.glob(os.path.join(fix_dir, "layer*.pt")))
    if not fix_paths:
        print("  (no fixtures found -- run capture_fixtures.py first)")
    for p in fix_paths:
        ok, _, _ = run_fixture(p)
        all_ok &= ok

    print("\n== latency (single layer, chunk=64) ==")
    print(f"{'S':>6} {'B':>2} {'l2':>2} | {'fast(ms)':>10} {'fallback(ms)':>13} {'speedup':>8}")
    for S, B, l2tag, mf, mr in rows:
        print(f"{S:6d} {B:2d} {l2tag:>2} | {mf:10.3f} {mr:13.3f} {mr / mf:7.1f}x")

    print(f"\nRESULT: {'ALL PASS' if all_ok else 'FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
