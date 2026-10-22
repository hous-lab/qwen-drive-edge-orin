#!/usr/bin/env python3
"""Chain gate v3 (2026-10-06, team-lead ruling): aggregate-blocking, scene-recording.

BLOCKING (all must hold):
  - aggregate mean|delta| <= 5 cm
  - ADE5 relative delta <= +10% (degradation bound)
  - FDE5 relative delta <= +10%
  - no NaN/Inf in engine trajectories
RECORD (archived, non-blocking):
  - per-scene mean|delta| and per-scene cos (full tables)
  - review list: scenes with per-scene mean|delta| > 5 cm

Rationale: every per-scene cos miss observed across all arms is a degenerate/
ill-conditioned short-norm plan with near-zero trajectory delta (worst 0.843cm
against the 5cm bar); deployment semantics are governed by aggregate ADE and
relative deviation from baseline, not per-scene cos.
"""
import json
import sys

import numpy as np

GATE_TRAJ_CM = 5.0
GATE_REL_PCT = 10.0
REVIEW_CM = 5.0


def load(path):
    rows = {}
    for line in open(path):
        if "_manifest" in line:
            continue
        r = json.loads(line)
        rows[r.get("scene_token") or r["token"]] = r
    return rows


def main(base_path, eng_path, archive_path=None, name="arm"):
    base, eng = load(base_path), load(eng_path)
    common = [k for k in base if k in eng]
    if not common:
        sys.exit("no common scenes between the two runs")

    nan_hits = []
    deltas, coss, a = [], [], {"base": [], "eng": []}
    scene_rows = []
    for k in common:
        pb = np.asarray(base[k]["trajectories"], dtype=np.float64)
        pe = np.asarray(eng[k]["trajectories"], dtype=np.float64)
        if pb.ndim == 2:
            pb = pb[None]
        if pe.ndim == 2:
            pe = pe[None]
        if not np.all(np.isfinite(pe)):
            nan_hits.append(k)
        d = float(np.abs(pb - pe).mean()) * 100.0            # m -> cm
        c = float((pb * pe).sum() / max(np.linalg.norm(pb) * np.linalg.norm(pe), 1e-30))
        deltas.append(d)
        coss.append(c)
        scene_rows.append({"scene": k, "mean_abs_cm": round(d, 4),
                           "cos": round(c, 7)})
        gt = base[k].get("future_trajectory")
        if gt is not None:
            for tag, p in (("base", pb), ("eng", pe)):
                dd = np.linalg.norm(p[0, :, :2] - np.asarray(gt)[:, :2], axis=-1)
                a[tag].append((dd[:30].mean(), dd.mean(), dd[-1]))

    deltas, coss = np.asarray(deltas), np.asarray(coss)
    mean_d = float(deltas.mean())
    rel = {}
    if a["base"]:
        names = ("ADE3", "ADE5", "FDE5")
        bm, em = np.mean(a["base"], axis=0), np.mean(a["eng"], axis=0)
        for nm, b, e in zip(names, bm, em):
            rel[nm] = (e - b) / b * 100.0
    ade5 = rel.get("ADE5", 0.0)
    fde5 = rel.get("FDE5", 0.0)
    review = [r for r in scene_rows if r["mean_abs_cm"] > REVIEW_CM]

    print(f"[v3] {name}: scenes compared: {len(common)} (with GT: {len(a['base'])})")
    print(f"[v3] {name}: aggregate mean|delta| {mean_d:.3f} cm "
          f"(p90 {np.percentile(deltas, 90):.3f}, max {deltas.max():.3f})")
    print(f"[v3] {name}: ADE5 {ade5:+.2f}%  FDE5 {fde5:+.2f}%  "
          f"NaN/Inf scenes: {len(nan_hits)}")
    print(f"[v3] {name}: cos min {coss.min():.7f} (recorded, non-blocking); "
          f"review list (>5cm scenes): {len(review)}")

    blocking = (mean_d <= GATE_TRAJ_CM
                and ade5 <= GATE_REL_PCT and fde5 <= GATE_REL_PCT
                and not nan_hits)
    print(f"[v3] {name}: gate mean|d|<={GATE_TRAJ_CM}cm AND ADE5/FDE5<="
          f"+{GATE_REL_PCT}% AND no-NaN: {'PASS' if blocking else 'FAIL'}")
    print(f"VERDICT-V3[{name}]:", "PASS" if blocking else "FAIL")

    if archive_path:
        with open(archive_path, "w") as f:
            json.dump({
                "gate": "v3", "arm": name,
                "blocking": {"mean_abs_cm": mean_d,
                             "ade5_rel_pct": ade5, "fde5_rel_pct": fde5,
                             "nan_scenes": nan_hits,
                             "thresholds": {"mean_cm": GATE_TRAJ_CM,
                                            "rel_pct": GATE_REL_PCT}},
                "verdict": "PASS" if blocking else "FAIL",
                "record": {"cos_min": float(coss.min()),
                           "cos_mean": float(coss.mean()),
                           "review_scenes_over_5cm": review,
                           "scenes": scene_rows},
            }, f, indent=1)
        print(f"[v3] {name}: archived -> {archive_path}")
    return 0 if blocking else 1


if __name__ == "__main__":
    name = sys.argv[4] if len(sys.argv) > 4 else "arm"
    sys.exit(main(sys.argv[1], sys.argv[2],
                  sys.argv[3] if len(sys.argv) > 3 else None, name))
