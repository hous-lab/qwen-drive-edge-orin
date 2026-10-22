#!/usr/bin/env python3
"""Chain gate: trajectory agreement between two runs of the SAME scenes.

Criteria (task spec): per-scene flattened trajectory cos >= 0.999 AND
mean|delta| <= 5 cm across scenes. ADE vs GT reported for context when both
runs carry future_trajectory. Mirrors eval_engine_inloop.py's ADE logic.
"""
import json
import sys

import numpy as np

GATE_TRAJ_CM = 5.0
GATE_COS = 0.999


def load(path):
    rows = {}
    for line in open(path):
        if "_manifest" in line:
            continue
        r = json.loads(line)
        rows[r.get("scene_token") or r["token"]] = r
    return rows


def main(base_path, eng_path):
    base, eng = load(base_path), load(eng_path)
    common = [k for k in base if k in eng]
    if not common:
        sys.exit("no common scenes between the two runs")

    deltas, coss, worst = [], [], (None, 0.0, 1.0)
    a = {"base": [], "eng": []}
    for k in common:
        pb = np.asarray(base[k]["trajectories"], dtype=np.float64)
        pe = np.asarray(eng[k]["trajectories"], dtype=np.float64)
        if pb.ndim == 2:
            pb = pb[None]
        if pe.ndim == 2:
            pe = pe[None]
        d = float(np.abs(pb - pe).mean()) * 100.0            # m -> cm
        c = float((pb * pe).sum() / max(np.linalg.norm(pb) * np.linalg.norm(pe), 1e-30))
        deltas.append(d)
        coss.append(c)
        if d > worst[1]:
            worst = (k, d, c)
        gt = base[k].get("future_trajectory")
        if gt is not None:
            for tag, p in (("base", pb), ("eng", pe)):
                dd = np.linalg.norm(p[0, :, :2] - np.asarray(gt)[:, :2], axis=-1)
                a[tag].append((dd[:30].mean(), dd.mean(), dd[-1]))

    deltas, coss = np.asarray(deltas), np.asarray(coss)
    print(f"scenes compared: {len(common)} (with GT: {len(a['base'])})")
    print(f"traj mean|delta| : mean {deltas.mean():.3f} cm, "
          f"p90 {np.percentile(deltas, 90):.3f} cm, max {deltas.max():.3f} cm")
    print(f"traj cos         : min {coss.min():.7f}, mean {coss.mean():.7f}")
    print(f"worst scene      : {worst[0]} ({worst[1]:.3f} cm, cos {worst[2]:.6f})")
    if a["base"]:
        names = ("ADE3", "ADE5", "FDE5")
        bm, em = np.mean(a["base"], axis=0), np.mean(a["eng"], axis=0)
        for nm, b, e in zip(names, bm, em):
            print(f"{nm:6s} {b:8.3f} -> {e:8.3f}  ({(e - b) / b * 100:+.2f}%)")

    ok = deltas.mean() <= GATE_TRAJ_CM and coss.min() >= GATE_COS
    print(f"gate mean|delta| <= {GATE_TRAJ_CM}cm and cos >= {GATE_COS}: "
          f"{'PASS' if ok else 'FAIL'}")
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
