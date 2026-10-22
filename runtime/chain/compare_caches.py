#!/usr/bin/env python3
"""Compare cache dumps from the torch arm vs the engine arm (chain bisect)."""
import sys

import numpy as np


def main(dir_a, dir_b):
    for scene in ("cache_0000", "cache_0001"):
        try:
            a = np.load(f"{dir_a}/{scene}.npz")
            b = np.load(f"{dir_b}/{scene}.npz")
        except FileNotFoundError:
            break
        print(f"== {scene} ==")
        if "anchor" in a and "anchor" in b:
            an = a["anchor"].astype(np.int64)
            bn = b["anchor"].astype(np.int64)
            print(f"anchor equal: {np.array_equal(an, bn)} "
                  f"(a[:6]={an[:6].tolist()} b[:6]={bn[:6].tolist()})")
        for g in range(8):
            ka, va = a[f"k_{g}"].astype(np.float64), a[f"v_{g}"].astype(np.float64)
            kb, vb = b[f"k_{g}"].astype(np.float64), b[f"v_{g}"].astype(np.float64)
            if ka.shape != kb.shape:
                print(f"g{g}: SHAPE {ka.shape} vs {kb.shape}")
                continue
            d_k = np.abs(ka - kb)
            d_v = np.abs(va - vb)
            ck = (ka * kb).sum() / max(np.linalg.norm(ka) * np.linalg.norm(kb), 1e-30)
            cv = (va * vb).sum() / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-30)
            print(f"g{g}: k |d| mean {d_k.mean():.4e} max {d_k.max():.4e} cos {ck:.6f} | "
                  f"v |d| mean {d_v.mean():.4e} max {d_v.max():.4e} cos {cv:.6f}")
    # trajectories of the debug runs
    import json
    for name in ("dbg_torch", "dbg_eng"):
        rows = [json.loads(l) for l in open(f"chain_out/{name}.jsonl")
                if "_manifest" not in l]
        for r in rows:
            p = np.asarray(r["trajectories"], dtype=np.float64)
            print(f"{name} {r.get('scene_token','?')[:8]} traj[:3]: "
                  f"{np.round(p.reshape(-1, 3)[:3], 3).tolist()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
