#!/usr/bin/env python3
"""M-RoPE composite rope-table rows for the Edge-LLM text engine.

The raw engine consumes ONE static table ``rope_rotary_cos_sin [1, kvLen, 64]``
fp32; row layout is ``cos(angle_0..31) || sin(angle_0..31)`` (validated by the
tiny + full gates with a plain arange table). Production prompts carry vision
tokens whose true rotary positions are the 3-component M-RoPE values from
``Qwen3_5Model.get_rope_index`` -> position_ids [3, B, S].

Qwen3.5 interleaved M-RoPE (transformers modeling_qwen3_5):
    freqs[c, j] = position_ids[c, tok] * inv_freq[j]      (c in T/H/W, j<32)
    interleaved layout: frequency j belongs to component j % 3
        (0 -> T, 1 -> H, 2 -> W; mrope_section [11, 11, 10])
    emb = cat(freqs_selected, freqs_selected)  -> cos/sin over 64 dims

Because each token sits at a unique row (scalar position = its sequence
index, kvcache_start_index = 0), the composite row for token t can simply be
WRITTEN into table[t] -- the engine's row lookup by position then applies the
exact per-token M-RoPE. This module builds those rows in numpy and
self-tests bit-comparable against the torch rotary module.
"""
import numpy as np

ROPE_THETA = 1e7
ROTARY_DIM = 64          # head_dim 256 * partial_rotary_factor 0.25
HALF = ROTARY_DIM // 2   # 32 frequency pairs
MROPE_SECTION = (11, 11, 10)


def inv_freq_table(theta=ROPE_THETA, rotary_dim=ROTARY_DIM):
    """inv_freq[j] = theta^(-2j/rotary_dim), j < rotary_dim/2 (matches HF)."""
    j = np.arange(0, rotary_dim, 2, dtype=np.float64)
    return theta ** (-j / rotary_dim)


def component_map(section=MROPE_SECTION):
    """Frequency index -> M-RoPE component (0=T, 1=H, 2=W), interleaved."""
    comp = np.zeros(HALF, dtype=np.int64)
    idx = 0
    for c, n in enumerate(section):
        comp[idx:idx + n] = c
        idx += n
    assert idx == HALF
    # interleaved remap: T keeps slots j%3==0, H gets j%3==1, W gets j%3==2
    inter = np.empty(HALF, dtype=np.int64)
    counts = [0, 0, 0]
    for j in range(HALF):
        c = j % 3
        inter[j] = c
        counts[c] += 1
    assert counts == list(section), (counts, section)
    return inter


def mrope_rows(position_ids, theta=ROPE_THETA, rotary_dim=ROTARY_DIM):
    """Build engine rope rows for one sequence.

    position_ids : [3, S] array-like of per-token (T, H, W) positions.
    returns      : [S, rotary_dim] float32, row = cos || sin.
    """
    pos = np.asarray(position_ids, dtype=np.float64)
    assert pos.shape[0] == 3, pos.shape
    s = pos.shape[1]
    inv = inv_freq_table(theta, rotary_dim)             # [32]
    comp = component_map()                              # [32]
    ang = pos[comp] * inv[:, None]                      # [32, S]
    rows = np.concatenate([np.cos(ang), np.sin(ang)], axis=0).T  # [S, 64]
    # C-contiguity is LOAD-BEARING: the engine reads the raw buffer row-major,
    # so an F-contiguous (transpose-view) array would be consumed scrambled
    # even though every torch/numpy consumer sees the right logical values
    # (bisect 2026-10-06: this exact bug cost the KV gate 0.9998 -> 0.66).
    return np.ascontiguousarray(rows, dtype=np.float32)


def arange_rows(n, theta=ROPE_THETA, rotary_dim=ROTARY_DIM):
    """Plain 1D table (gate reference): row p = rope(p), all three comps = p."""
    return mrope_rows(np.tile(np.arange(n, dtype=np.float64), (3, 1)),
                      theta, rotary_dim)


def selftest():
    """Rows must match transformers' Qwen3_5TextRotaryEmbedding output."""
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    class _Cfg:
        rope_parameters = {"rope_type": "default", "rope_theta": ROPE_THETA,
                           "partial_rotary_factor": 0.25}
        max_position_embeddings = 4096
        rope_theta = ROPE_THETA
        head_dim = 256
        mrope_section = list(MROPE_SECTION)

    rot = Qwen3_5TextRotaryEmbedding(_Cfg())
    rng = np.random.default_rng(0)
    s = 97
    # image-like block structure: T constant inside a block, H/W per row
    t = np.repeat(np.arange(7), 14)[:s].astype(np.float64)
    h = np.tile(np.arange(14), 7)[:s].astype(np.float64)
    w = np.tile(np.arange(14) * 3, 7)[:s].astype(np.float64)
    pos = np.stack([t, h, w])                            # [3, s]
    x = torch.zeros(1, s, 8)
    cos_t, sin_t = rot(x, torch.from_numpy(pos[:, None, :]).long())
    mine = mrope_rows(pos)
    # torch cos/sin are each [s,64] with duplicated halves (emb=cat(freqs,freqs));
    # the engine row is cos(angle_0..31) || sin(angle_0..31) = cos[:32] || sin[32:]
    ref = np.concatenate([cos_t[0, :, :32].numpy(), sin_t[0, :, 32:].numpy()],
                         axis=-1)
    err = np.abs(mine.astype(np.float64) - ref.astype(np.float64)).max()
    print(f"mrope rows vs torch: max|d| = {err:.3e} over [{s},64]")
    assert err < 1e-5, "mrope row builder diverges from torch rotary"
    assert mine.flags["C_CONTIGUOUS"], "mrope rows must stay C-contiguous"
    # arange degenerate case must equal the gate script's table
    inv = ROPE_THETA ** (-np.arange(0, ROTARY_DIM, 2, dtype=np.float64) / ROTARY_DIM)
    ang = np.arange(16, dtype=np.float64)[:, None] * inv[None, :]
    gate = np.concatenate([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)
    err2 = np.abs(arange_rows(16) - gate).max()
    print(f"arange rows vs gate formula: max|d| = {err2:.3e}")
    assert err2 < 1e-5
    print("SELFTEST PASS")


if __name__ == "__main__":
    selftest()
