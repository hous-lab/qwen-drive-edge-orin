"""Self-written chunked gated delta rule (Triton) for Jetson Orin (triton 3.5.1+).

Drop-in replacement for transformers 5.14.1 qwen3_5 torch_chunk_gated_delta_rule
(the pure-torch fallback used when fla is unavailable). Math is aligned
term-by-term with modeling_qwen3_5.py::torch_chunk_gated_delta_rule (line 248):

  per chunk (g cumsumed inside the chunk -> g_cum):
    decay[i,j] = exp(g_cum[i] - g_cum[j])   for i >= j, else 0        (L291)
    A          = -(k*beta) @ k^T * decay, strict lower triangular     (L292)
    Tinv       = (I - A)^{-1}                                         (L293-297)
    w          = Tinv @ (v*beta)                                      (L298)
    u          = Tinv @ (k*beta*exp(g_cum))                           (L299)
    v_new      = w - u @ S                                            (L312-313)
    o          = (q*exp(g_cum)) @ S + (q @ k^T * decay) @ v_new       (L311,314-315)
    S          = S * exp(g_cum[-1]) + (k*exp(g_cum[-1]-g_cum))^T @ v_new  (L316-319)

Layout/dtype contract (identical to the fallback as called by qwen3_5):
  q,k  [B,T,HV,DK] bf16  -- the caller repeat_interleave's HK->HV (L524-525)
                           before calling, so the kernel sees HV key heads too.
  v    [B,T,HV,DV] bf16
  g,b  [B,T,HV]    fp32 / float
  o    [B,T,HV,DV] bf16,  final_state [B,HV,DK,DV] fp32.

Implementation notes:
  - two-kernel scheme (fla-style): K1 "prep" is parallel over (B, HV, NC) and
    does all chunk-local heavy math (A, the (I-A)^{-1} inversion, w, u) into a
    fp32 scratch; K2 "scan" is the inherently serial chunk walk per (B, HV)
    with only 5 dots per chunk left. A single fused kernel measured 5-10x
    slower on Orin: one block then serially executes ~20 small dots per chunk
    and per-dot operand-staging latency dominates.
  - the scan loop contains NO transcendentals and no cumsum: prep also stores
    decay, exp(g_cum), exp(g_last-g_cum) and exp(g_last) per chunk. On Orin a
    tl.cumsum is a cross-warp reduction (~20us/step) and would sit on the
    serial critical path NC=50 times.
  - Tinv = (I-A)^{-1} is built blockwise: each 16x16 diagonal leaf inverts its
    own strict-lower block via 4 squaring steps (T <- T + T@P, P <- P@P), then
    10 blockwise forward-substitution dots assemble the 64x64 T. The Phase-1
    full-64x64 doubling formed A^32/A^64 intermediates up to ~1e18 where tf32
    quantization explodes -> NaN on real, correlated activations (max|k_i.k_j|
    = 1.0000, g cumsum to -1.7e4 on the layer00 fixture). 16x16 leaf powers
    stay <= ~1e3-1e4 -- safe.
  - all tl.dot calls take fp32 operands with an explicit input_precision
    class; see the "Dot precision" block below for the five classes and why
    each defaults where it does. NO make_block_ptr (NaNs on this
    aarch64/triton combo); no tl.static_range for long loops.
  - Tail chunk is handled by masking loads with 0.0, which exactly reproduces
    the fallback's F.pad semantics (padded beta/g = 0 keep cumsum constant and
    zero out k_beta/v_beta rows).
"""
import os

import torch
import triton
import triton.language as tl

# Dot precision, five classes (shipped default = the "R5" config that passed
# the model-level G0 regression: KV cos >= 0.9996 on all 16 dumped KV tensors,
# 0 NaN, trajectory cos 0.99997+; see deltanet_kernel/g0_metrics.py):
#   fine   (_DOT_PREC): inversion leaves + blockwise combine -- these form
#          (I-A)^{-1}, whose rounding compounds into the recurrent state over
#          NC chunks x 24 layers. Default "tf32x3" (~fp32 accuracy).
#          DELTANET_PREC overrides.
#   A      (_APREC): A = -(k*beta)@k^T; its rounding perturbs T = (I-A)^-1
#          which passes into u/w -> state even when the u/w dots are x3.
#          Default "tf32x3" (tf32 here measured layer00 fixture state rel
#          1.5e-3 vs the 1e-3 drift budget; x3 gives 1.1e-5). DELTANET_A_PREC.
#   uw     (_UW): u/w = T@(...) feed v_new -> state every chunk.
#          Default "tf32x3". DELTANET_UW_PREC.
#   scan state (_SCANPREC): the scan's two STATE dots (u@S and
#          k_dec^T@v_new) rewrite the recurrent state every chunk; tf32 there
#          alone measured 3.6e-3..1e-2 fixture state rel. Default "tf32x3".
#          DELTANET_SCAN_PREC.
#   coarse (_COARSE): ONLY the scan's three terminal o dots (q@k^T*de,
#          q*eg@S, M@v_new) -- no compounding. Default "tf32": x3 here is
#          catastrophically slow on Orin (scan 5.8ms -> 71-89ms, whole kernel
#          ~120ms ~= the torch fallback) for a bf16-invisible gain: o is bf16
#          (1 ulp ~ 3.9e-3 rel), tf32 keeps o tensor-L2 rel at ~2e-3 with
#          cos >= 0.99999. DELTANET_COARSE_PREC.
_FINE_STR = os.environ.get("DELTANET_PREC", "tf32x3")
_COARSE_STR = os.environ.get("DELTANET_COARSE_PREC", "tf32")
_SCAN_STR = os.environ.get("DELTANET_SCAN_PREC", "tf32x3")
_UW_STR = os.environ.get("DELTANET_UW_PREC", "tf32x3")
_A_STR = os.environ.get("DELTANET_A_PREC", "tf32x3")
_DOT_PREC = tl.constexpr(_FINE_STR)
_COARSE = tl.constexpr(_COARSE_STR)
_SCANPREC = tl.constexpr(_SCAN_STR)
_UW = tl.constexpr(_UW_STR)
_APREC = tl.constexpr(_A_STR)


@triton.jit
def _leaf_inv(A):
    # (I - A)^{-1} for a strict-lower 16x16 block: A^16 == 0, so
    # T = sum_{i<16} A^i via 4 squaring steps. Intermediate powers stay
    # <= C(15,8)*max|A|^15 ~ 1e3-1e4 -- safe in fp32/tf32x3. (The previous
    # full-64x64 doubling formed A^32/A^64 intermediates up to ~1e18 where
    # tf32 quantization explodes -> NaN on real, correlated activations.)
    rr = tl.arange(0, 16)
    eye = tl.where(rr[:, None] == rr[None, :], 1.0, 0.0)
    P = A
    Tm = eye
    for _ in range(4):
        Tm = Tm + tl.dot(Tm, P, input_precision=_DOT_PREC)
        P = tl.dot(P, P, input_precision=_DOT_PREC)
    return Tm


@triton.jit
def _chunk_prep_kernel(
    k_ptr, v_ptr, g_ptr, b_ptr, w_ptr, u_ptr,
    de_ptr, eg_ptr, ed_ptr, gl_ptr,
    T, NC,
    skb, skt, skh, skd,
    svb, svt, svh, svd,
    sgb, sgt, sgh,
    sbb, sbt, sbh,
    swb, swh, swc, swt, swd,
    sub, suh, suc, sut, sud,
    sdeb, sdeh, sdec, sdet, sded,
    segb, segh, segc, segt,
    sedb, sedh, sedc, sedt,
    sglb, sglh, sglc,
    DK: tl.constexpr, DV: tl.constexpr, CS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_c = tl.program_id(2)

    rk = tl.arange(0, DK)
    rv = tl.arange(0, DV)
    r = tl.arange(0, 16)

    kb = k_ptr + pid_b * skb + pid_h * skh
    vb = v_ptr + pid_b * svb + pid_h * svh
    gb = g_ptr + pid_b * sgb + pid_h * sgh
    bb = b_ptr + pid_b * sbb + pid_h * sbh

    # four 16-row blocks; tail rows masked with 0 (== the fallback's F.pad)
    tofs0 = pid_c * CS + r
    tofs1 = tofs0 + 16
    tofs2 = tofs0 + 32
    tofs3 = tofs0 + 48
    m0 = tofs0 < T
    m1 = tofs1 < T
    m2 = tofs2 < T
    m3 = tofs3 < T

    g0 = tl.load(gb + tofs0 * sgt, mask=m0, other=0.0)
    g1 = tl.load(gb + tofs1 * sgt, mask=m1, other=0.0)
    g2 = tl.load(gb + tofs2 * sgt, mask=m2, other=0.0)
    g3 = tl.load(gb + tofs3 * sgt, mask=m3, other=0.0)
    b0 = tl.load(bb + tofs0 * sbt, mask=m0, other=0.0)
    b1 = tl.load(bb + tofs1 * sbt, mask=m1, other=0.0)
    b2 = tl.load(bb + tofs2 * sbt, mask=m2, other=0.0)
    b3 = tl.load(bb + tofs3 * sbt, mask=m3, other=0.0)

    # chunk cumsum carried across the four blocks; g_last == c3's last row
    c0 = tl.cumsum(g0, axis=0)
    c1 = tl.sum(tl.where(r == 15, c0, 0.0)) + tl.cumsum(g1, axis=0)
    c2 = tl.sum(tl.where(r == 15, c1, 0.0)) + tl.cumsum(g2, axis=0)
    c3 = tl.sum(tl.where(r == 15, c2, 0.0)) + tl.cumsum(g3, axis=0)
    g_last = tl.sum(tl.where(r == 15, c3, 0.0))

    kf0 = tl.load(kb + tofs0[:, None] * skt + rk[None, :] * skd, mask=m0[:, None], other=0.0).to(tl.float32)
    kf1 = tl.load(kb + tofs1[:, None] * skt + rk[None, :] * skd, mask=m1[:, None], other=0.0).to(tl.float32)
    kf2 = tl.load(kb + tofs2[:, None] * skt + rk[None, :] * skd, mask=m2[:, None], other=0.0).to(tl.float32)
    kf3 = tl.load(kb + tofs3[:, None] * skt + rk[None, :] * skd, mask=m3[:, None], other=0.0).to(tl.float32)

    # decay blocks: diag LOWER-INCLUSIVE (i>=j, diagonal = 1 -- the scan's
    # M = q@k^T*de term needs it, fallback L311), off-diag (i>j) full.
    # exp args <= 0 so no overflow.
    D00 = tl.where(r[:, None] >= r[None, :], tl.exp(c0[:, None] - c0[None, :]), 0.0)
    D11 = tl.where(r[:, None] >= r[None, :], tl.exp(c1[:, None] - c1[None, :]), 0.0)
    D22 = tl.where(r[:, None] >= r[None, :], tl.exp(c2[:, None] - c2[None, :]), 0.0)
    D33 = tl.where(r[:, None] >= r[None, :], tl.exp(c3[:, None] - c3[None, :]), 0.0)
    D10 = tl.exp(c1[:, None] - c0[None, :])
    D20 = tl.exp(c2[:, None] - c0[None, :])
    D21 = tl.exp(c2[:, None] - c1[None, :])
    D30 = tl.exp(c3[:, None] - c0[None, :])
    D31 = tl.exp(c3[:, None] - c1[None, :])
    D32 = tl.exp(c3[:, None] - c2[None, :])

    # A blocks = -(k*beta) @ k^T * decay (fallback L292). Diag blocks are
    # re-masked STRICT lower (the decay diag=1 must not leak into A); off-diag
    # blocks are all-strict by construction (block row > block col).
    A00 = tl.where(r[:, None] > r[None, :],
                   -tl.dot(kf0 * b0[:, None], tl.trans(kf0), input_precision=_APREC) * D00, 0.0)
    A11 = tl.where(r[:, None] > r[None, :],
                   -tl.dot(kf1 * b1[:, None], tl.trans(kf1), input_precision=_APREC) * D11, 0.0)
    A22 = tl.where(r[:, None] > r[None, :],
                   -tl.dot(kf2 * b2[:, None], tl.trans(kf2), input_precision=_APREC) * D22, 0.0)
    A33 = tl.where(r[:, None] > r[None, :],
                   -tl.dot(kf3 * b3[:, None], tl.trans(kf3), input_precision=_APREC) * D33, 0.0)
    A10 = -tl.dot(kf1 * b1[:, None], tl.trans(kf0), input_precision=_APREC) * D10
    A20 = -tl.dot(kf2 * b2[:, None], tl.trans(kf0), input_precision=_APREC) * D20
    A21 = -tl.dot(kf2 * b2[:, None], tl.trans(kf1), input_precision=_APREC) * D21
    A30 = -tl.dot(kf3 * b3[:, None], tl.trans(kf0), input_precision=_APREC) * D30
    A31 = -tl.dot(kf3 * b3[:, None], tl.trans(kf1), input_precision=_APREC) * D31
    A32 = -tl.dot(kf3 * b3[:, None], tl.trans(kf2), input_precision=_APREC) * D32

    # (I-A)^{-1} = [[T00, 0, 0, 0], [T10, T11, 0, 0], [T20, T21, T22, 0],
    #               [T30, T31, T32, T33]] via leaves + forward substitution
    T00 = _leaf_inv(A00)
    T11 = _leaf_inv(A11)
    T22 = _leaf_inv(A22)
    T33 = _leaf_inv(A33)

    T10 = tl.dot(T11, tl.dot(A10, T00, input_precision=_DOT_PREC), input_precision=_DOT_PREC)
    T20 = tl.dot(T22, tl.dot(A20, T00, input_precision=_DOT_PREC)
                 + tl.dot(A21, T10, input_precision=_DOT_PREC), input_precision=_DOT_PREC)
    T21 = tl.dot(T22, tl.dot(A21, T11, input_precision=_DOT_PREC), input_precision=_DOT_PREC)
    T30 = tl.dot(T33, tl.dot(A30, T00, input_precision=_DOT_PREC)
                 + tl.dot(A31, T10, input_precision=_DOT_PREC)
                 + tl.dot(A32, T20, input_precision=_DOT_PREC), input_precision=_DOT_PREC)
    T31 = tl.dot(T33, tl.dot(A31, T11, input_precision=_DOT_PREC)
                 + tl.dot(A32, T21, input_precision=_DOT_PREC), input_precision=_DOT_PREC)
    T32 = tl.dot(T33, tl.dot(A32, T22, input_precision=_DOT_PREC), input_precision=_DOT_PREC)

    # u_i = sum_{j<=i} T_ij @ (k_j*beta_j*exp(c_j))  (fallback L299)
    e0 = kf0 * (b0 * tl.exp(c0))[:, None]
    e1 = kf1 * (b1 * tl.exp(c1))[:, None]
    e2 = kf2 * (b2 * tl.exp(c2))[:, None]
    e3 = kf3 * (b3 * tl.exp(c3))[:, None]
    u0 = tl.dot(T00, e0, input_precision=_UW)
    u1 = tl.dot(T10, e0, input_precision=_UW) + tl.dot(T11, e1, input_precision=_UW)
    u2 = tl.dot(T20, e0, input_precision=_UW) + tl.dot(T21, e1, input_precision=_UW) \
        + tl.dot(T22, e2, input_precision=_UW)
    u3 = tl.dot(T30, e0, input_precision=_UW) + tl.dot(T31, e1, input_precision=_UW) \
        + tl.dot(T32, e2, input_precision=_UW) + tl.dot(T33, e3, input_precision=_UW)

    # scratch is [.., NC, CS, ..]: the chunk offset goes in ONCE via pid_c*suc,
    # so stores must use LOCAL in-chunk rows (r, 16+r, 32+r, 48+r) -- the
    # global tofs* would double-count the chunk offset and write out of bounds.
    uo = u_ptr + pid_b * sub + pid_h * suh + pid_c * suc
    tl.store(uo + r[:, None] * sut + rk[None, :] * sud, u0)
    tl.store(uo + (16 + r)[:, None] * sut + rk[None, :] * sud, u1)
    tl.store(uo + (32 + r)[:, None] * sut + rk[None, :] * sud, u2)
    tl.store(uo + (48 + r)[:, None] * sut + rk[None, :] * sud, u3)

    # w_i = sum_{j<=i} T_ij @ (v_j*beta_j)  (fallback L298); v loaded after the
    # u blocks so kf can retire first (register pressure)
    vf0 = tl.load(vb + tofs0[:, None] * svt + rv[None, :] * svd, mask=m0[:, None], other=0.0).to(tl.float32)
    vf1 = tl.load(vb + tofs1[:, None] * svt + rv[None, :] * svd, mask=m1[:, None], other=0.0).to(tl.float32)
    vf2 = tl.load(vb + tofs2[:, None] * svt + rv[None, :] * svd, mask=m2[:, None], other=0.0).to(tl.float32)
    vf3 = tl.load(vb + tofs3[:, None] * svt + rv[None, :] * svd, mask=m3[:, None], other=0.0).to(tl.float32)
    w0 = tl.dot(T00, vf0 * b0[:, None], input_precision=_UW)
    w1 = tl.dot(T10, vf0 * b0[:, None], input_precision=_UW) + tl.dot(T11, vf1 * b1[:, None], input_precision=_UW)
    w2 = tl.dot(T20, vf0 * b0[:, None], input_precision=_UW) + tl.dot(T21, vf1 * b1[:, None], input_precision=_UW) \
        + tl.dot(T22, vf2 * b2[:, None], input_precision=_UW)
    w3 = tl.dot(T30, vf0 * b0[:, None], input_precision=_UW) + tl.dot(T31, vf1 * b1[:, None], input_precision=_UW) \
        + tl.dot(T32, vf2 * b2[:, None], input_precision=_UW) + tl.dot(T33, vf3 * b3[:, None], input_precision=_UW)

    wo = w_ptr + pid_b * swb + pid_h * swh + pid_c * swc
    tl.store(wo + r[:, None] * swt + rv[None, :] * swd, w0)
    tl.store(wo + (16 + r)[:, None] * swt + rv[None, :] * swd, w1)
    tl.store(wo + (32 + r)[:, None] * swt + rv[None, :] * swd, w2)
    tl.store(wo + (48 + r)[:, None] * swt + rv[None, :] * swd, w3)

    # eg/ed/gl: exp terms precomputed so the serial scan loop has no transcendentals
    tl.store(eg_ptr + pid_b * segb + pid_h * segh + pid_c * segc + r * segt, tl.exp(c0))
    tl.store(eg_ptr + pid_b * segb + pid_h * segh + pid_c * segc + (16 + r) * segt, tl.exp(c1))
    tl.store(eg_ptr + pid_b * segb + pid_h * segh + pid_c * segc + (32 + r) * segt, tl.exp(c2))
    tl.store(eg_ptr + pid_b * segb + pid_h * segh + pid_c * segc + (48 + r) * segt, tl.exp(c3))
    tl.store(ed_ptr + pid_b * sedb + pid_h * sedh + pid_c * sedc + r * sedt, tl.exp(g_last - c0))
    tl.store(ed_ptr + pid_b * sedb + pid_h * sedh + pid_c * sedc + (16 + r) * sedt, tl.exp(g_last - c1))
    tl.store(ed_ptr + pid_b * sedb + pid_h * sedh + pid_c * sedc + (32 + r) * sedt, tl.exp(g_last - c2))
    tl.store(ed_ptr + pid_b * sedb + pid_h * sedh + pid_c * sedc + (48 + r) * sedt, tl.exp(g_last - c3))
    tl.store(gl_ptr + pid_b * sglb + pid_h * sglh + pid_c * sglc, tl.exp(g_last))

    # decay blocks into the zero-initialized [.., CS, CS] scratch (upper blocks
    # stay zero); consumed by the scan kernel
    db = de_ptr + pid_b * sdeb + pid_h * sdeh + pid_c * sdec
    tl.store(db + r[:, None] * sdet + r[None, :] * sded, D00)
    tl.store(db + (16 + r)[:, None] * sdet + (16 + r)[None, :] * sded, D11)
    tl.store(db + (32 + r)[:, None] * sdet + (32 + r)[None, :] * sded, D22)
    tl.store(db + (48 + r)[:, None] * sdet + (48 + r)[None, :] * sded, D33)
    tl.store(db + (16 + r)[:, None] * sdet + r[None, :] * sded, D10)
    tl.store(db + (32 + r)[:, None] * sdet + r[None, :] * sded, D20)
    tl.store(db + (32 + r)[:, None] * sdet + (16 + r)[None, :] * sded, D21)
    tl.store(db + (48 + r)[:, None] * sdet + r[None, :] * sded, D30)
    tl.store(db + (48 + r)[:, None] * sdet + (16 + r)[None, :] * sded, D31)
    tl.store(db + (48 + r)[:, None] * sdet + (32 + r)[None, :] * sded, D32)


@triton.jit
def _chunk_scan_kernel(
    q_ptr, k_ptr, w_ptr, u_ptr, de_ptr, eg_ptr, ed_ptr, gl_ptr,
    o_ptr, s0_ptr, sf_ptr,
    T, NC, scale,
    sqb, sqt, sqh, sqd,
    skb, skt, skh, skd,
    swb, swh, swc, swt, swd,
    sub, suh, suc, sut, sud,
    sdeb, sdeh, sdec, sdet, sded,
    segb, segh, segc, segt,
    sedb, sedh, sedc, sedt,
    sglb, sglh, sglc,
    sob, sot, soh, sod,
    s0b, s0h, s0k, s0v,
    sfb, sfh, sfk, sfv,
    DK: tl.constexpr, DV: tl.constexpr, CS: tl.constexpr, BV: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_v = tl.program_id(2)  # DV slice id; S columns are independent across slices

    rk = tl.arange(0, DK)
    rv = pid_v * BV + tl.arange(0, BV)
    rt = tl.arange(0, CS)

    qb = q_ptr + pid_b * sqb + pid_h * sqh
    kb = k_ptr + pid_b * skb + pid_h * skh
    deb = de_ptr + pid_b * sdeb + pid_h * sdeh
    egb = eg_ptr + pid_b * segb + pid_h * segh
    edb = ed_ptr + pid_b * sedb + pid_h * sedh
    glb = gl_ptr + pid_b * sglb + pid_h * sglh
    wb = w_ptr + pid_b * swb + pid_h * swh
    ub = u_ptr + pid_b * sub + pid_h * suh

    # recurrent state slice S [DK, BV], fp32, lives in registers across chunks
    S = tl.load(s0_ptr + pid_b * s0b + pid_h * s0h + rk[:, None] * s0k + rv[None, :] * s0v)

    for c in range(0, NC):
        tofs = c * CS + rt
        tmask = tofs < T

        # all decay/exp terms precomputed by the parallel prep kernel
        de = tl.load(deb + c * sdec + rt[:, None] * sdet + rt[None, :] * sded)
        eg = tl.load(egb + c * segc + rt * segt)
        ed = tl.load(edb + c * sedc + rt * sedt)
        gl = tl.load(glb + c * sglc)

        w = tl.load(wb + c * swc + rt[:, None] * swt + rv[None, :] * swd)   # fp32 [CS,BV]
        u = tl.load(ub + c * suc + rt[:, None] * sut + rk[None, :] * sud)   # fp32 [CS,DK]
        v_new = w - tl.dot(u, S, input_precision=_SCANPREC)                     # [CS,BV] (L312-313)

        qf = tl.load(qb + tofs[:, None] * sqt + rk[None, :] * sqd,
                     mask=tmask[:, None], other=0.0).to(tl.float32) * scale
        kf = tl.load(kb + tofs[:, None] * skt + rk[None, :] * skd,
                     mask=tmask[:, None], other=0.0).to(tl.float32)

        M = tl.dot(qf, tl.trans(kf), input_precision=_COARSE) * de            # [CS,CS] diag kept (L311)
        o = tl.dot(qf * eg[:, None], S, input_precision=_COARSE) \
            + tl.dot(M, v_new, input_precision=_COARSE)                       # (L314-315)
        tl.store(o_ptr + pid_b * sob + pid_h * soh + tofs[:, None] * sot + rv[None, :] * sod,
                 o.to(o_ptr.dtype.element_ty), mask=tmask[:, None])

        # state update (L316-319); gl/ed come from prep and encode the padded-
        # tail semantics exactly (padding adds g=0 rows, matching the fallback)
        k_dec = kf * ed[:, None]
        S = S * gl + tl.dot(tl.trans(k_dec), v_new, input_precision=_SCANPREC)

    tl.store(sf_ptr + pid_b * sfb + pid_h * sfh + rk[:, None] * sfk + rv[None, :] * sfv, S)


def _l2norm(x, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
    return x * inv_norm


def chunk_gated_delta_rule_fast(
    query, key, value, g, beta,
    chunk_size: int = 64,
    initial_state=None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
):
    """Signature/layout-compatible replacement of torch_chunk_gated_delta_rule.

    query/key: [B,T,HV,DK], value: [B,T,HV,DV], g: [B,T,HV], beta: [B,T,HV].
    Returns (o [B,T,HV,DV] in input dtype, final_state [B,HV,DK,DV] fp32).
    """
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)

    B, T, HV, DK = key.shape
    DV = value.shape[-1]
    NC = triton.cdiv(T, chunk_size)

    qc = query.contiguous().to(torch.bfloat16)
    kc = key.contiguous().to(torch.bfloat16)
    vc = value.contiguous().to(torch.bfloat16)
    gc = g.to(torch.float32).contiguous()
    bc = beta.to(torch.float32).contiguous()

    o = torch.empty_like(vc)
    if initial_state is None:
        s0 = torch.zeros((B, HV, DK, DV), dtype=torch.float32, device=vc.device)
    else:
        s0 = initial_state.to(torch.float32).contiguous()
    sf = torch.empty((B, HV, DK, DV), dtype=torch.float32, device=vc.device)

    # fp32 scratch for per-chunk w/u (K1 -> K2 handoff)
    w = torch.empty((B, HV, NC, chunk_size, DV), dtype=torch.float32, device=vc.device)
    u = torch.empty((B, HV, NC, chunk_size, DK), dtype=torch.float32, device=vc.device)
    # precomputed decay / exp terms (K1 -> K2 handoff): keeps the serial scan
    # loop free of cumsum and transcendental ops. ZEROS because prep only
    # stores the lower+diag decay blocks; the upper triangle must read 0.
    de = torch.zeros((B, HV, NC, chunk_size, chunk_size), dtype=torch.float32, device=vc.device)
    eg = torch.empty((B, HV, NC, chunk_size), dtype=torch.float32, device=vc.device)
    ed = torch.empty_like(eg)
    gl = torch.empty((B, HV, NC), dtype=torch.float32, device=vc.device)

    _chunk_prep_kernel[(B, HV, NC)](
        kc, vc, gc, bc, w, u,
        de, eg, ed, gl,
        T, NC,
        *kc.stride(), *vc.stride(), *gc.stride(), *bc.stride(),
        *w.stride(), *u.stride(),
        *de.stride(), *eg.stride(), *ed.stride(), *gl.stride(),
        DK=DK, DV=DV, CS=chunk_size,
        num_warps=4, num_stages=1,
    )
    # scan: split the DV dimension across NSPLIT programs -- S column slices
    # are independent (u@S, qg@S, k_dec^T@v_new are all columnwise), q/k/M work
    # is redundantly recomputed per slice. tf32x3 staging grows SMEM ~1.5x vs
    # tf32, so BV=128 no longer fits either platform: Orin (166912B limit,
    # 196608B needed at BV=128) and the 4090 (101376B limit, 131072B needed at
    # BV=64) both run split=4/warps=4 (BV=32) -- measured best on Orin with
    # x3 state dots (5.64ms vs 9.14ms for split=2/warps=8, B=1 S=3200).
    if "orin" in torch.cuda.get_device_name(vc.device).lower():
        nsplit, scan_warps = 4, 4
    else:
        nsplit, scan_warps = 4, 4
    BV = DV // nsplit
    _chunk_scan_kernel[(B, HV, nsplit)](
        qc, kc, w, u, de, eg, ed, gl,
        o, s0, sf,
        T, NC, 1.0 / (DK ** 0.5),
        *qc.stride(), *kc.stride(),
        *w.stride(), *u.stride(),
        *de.stride(), *eg.stride(), *ed.stride(), *gl.stride(),
        *o.stride(), *s0.stride(), *sf.stride(),
        DK=DK, DV=DV, CS=chunk_size, BV=BV,
        num_warps=scan_warps, num_stages=1,
    )
    if not output_final_state:
        return o, None
    return o, sf
