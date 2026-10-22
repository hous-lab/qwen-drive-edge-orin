#!/usr/bin/env python3
"""PAI raw data -> Qwen-Drive scene jsonl adapter.

Reads PAI chunk0 clips (same pool as Alpamayo evalset_v2: first 94 valid clips,
t0 = 5.1s) and emits Qwen official scene contract:
  - 3 cameras x 4 timestamps (t-1.5/1.0/0.5/0s), frames saved as jpeg
  - hist_traj/vel/acc [16,3]/[16,2]/[16,2] @10Hz from official egomotion interpolator
  - future [50,3] @10Hz + valid mask (all valid: t0+5s < 20s clip)
  - ego_status 8 = [vx,vy,ax,ay, *driving_command(4)]
  - nav_command: derived from future heading change (0 straight / 1 left / 2 right)
  - instruction: official synthesis template (verified vs DrivingScene.instruction)

Output: <out>/qwen_scenes_<tag>.jsonl + <out>/frames/... (+ optional parquet pack)
"""
import argparse, json, math, os, sys
from pathlib import Path

import numpy as np
import scipy.spatial.transform as spt
import os
from PIL import Image

TRAIN_DIR = os.environ.get("QD_PAI_ROOT", ".")

CAM_NAMES = ["CAMERA_CROSS_LEFT_120FOV", "CAMERA_FRONT_WIDE_120FOV", "CAMERA_CROSS_RIGHT_120FOV"]
VIEW_TAGS = ["<FRONT LEFT VIEW>", "<FRONT VIEW>", "<FRONT RIGHT VIEW>"]
# camera order follows Qwen CAMERA_VIEWS order: FRONT, FRONT LEFT, FRONT RIGHT
CAMS_BY_VIEW = {
    "<FRONT VIEW>": "CAMERA_FRONT_WIDE_120FOV",
    "<FRONT LEFT VIEW>": "CAMERA_CROSS_LEFT_120FOV",
    "<FRONT RIGHT VIEW>": "CAMERA_CROSS_RIGHT_120FOV",
}
HIST_TS_OFFSETS = [-(1500), -(1000), -(500), 0]          # ms relative to t0, oldest first
TS_US = 1_000_000
N_HIST, N_FUT = 16, 50
DT_US = 100_000
CUR_W, CUR_H = 1280, 704      # 880 tokens after merge (matches roofline constants)
HIS_W, HIS_H = 384, 224       # 84 tokens
T0_US = 5_100_000
EOF_MARK = None


def yaw_from_rot(R):
    return math.atan2(R[1, 0], R[0, 0])


def to_local(xyz_w, quat_w, t0_xyz, t0_quat):
    """World -> t0 ego frame; returns xyz + yaw arrays."""
    r0 = spt.Rotation.from_quat(t0_quat)
    r0inv = r0.inv()
    xyz = r0inv.apply(np.asarray(xyz_w) - t0_xyz)
    yaw = (spt.Rotation.from_quat(np.asarray(quat_w)) * r0inv).as_euler("xyz")[:, 2]
    return xyz, (yaw + math.pi) % (2 * math.pi) - math.pi


_INSTRUCTION_HEADER = (
    "The input images are organized by camera view. Each view contains {num_frames} temporal "
    "frames captured at {interval:g}s intervals (frame 0 at {first_label}, frame {last_frame} is "
    "the current frame at t=0s).\n"
    "1. Historical trajectories (x, y, heading) in the current frame's ego coordinate system. "
    "Positive x points forward, positive y points left, and a positive heading indicates a left "
    "turn\uff1a\n"
)
_LABELS = ("t-1.5s", "t-1.0s", "t-0.5s", "t-0s")


def instruction_template(history, nav_command):
    """Replicates DrivingScene.instruction() synthesis exactly (4 frames, 1.5s/3)."""
    lines = "".join(
        " -{}: ({:.4f}, {:.4f}, {:.4f});\n".format(_LABELS[k], *history[k * 5])
        for k in range(4)
    )
    header = _INSTRUCTION_HEADER.format(
        num_frames=4, interval=0.5, first_label="t-1.5s", last_frame=3)
    command = ("GO STRAIGHT", "TURN LEFT", "TURN RIGHT")[nav_command]
    return f"{header}{lines}2. Active navigation command: [{command}]"


def derive_nav(future_yaw, straight_deg=15.0):
    """0 straight / 1 left / 2 right from net heading change over the 5s future."""
    dpsi = (future_yaw[-1] - future_yaw[0] + math.pi) % (2 * math.pi) - math.pi
    if abs(dpsi) < math.radians(straight_deg):
        return 0
    return 1 if dpsi > 0 else 2


def encode_clip(avdi, clip, sid, frames_dir):
    """One PAI clip -> (qwen scene record dict, img std check)."""
    import numpy as np

    t0 = T0_US
    hist_ts = t0 + np.arange(-(N_HIST - 1), 1) * DT_US
    fut_ts = t0 + np.arange(1, N_FUT + 1) * DT_US

    egomotion = avdi.get_clip_feature(clip, avdi.features.LABELS.EGOMOTION)
    eh = egomotion(hist_ts.astype(np.int64))
    ef = egomotion(fut_ts.astype(np.int64))

    h_xyz_w = np.asarray(eh.pose.translation)
    h_quat_w = np.asarray(eh.pose.rotation.as_quat())
    f_xyz_w = np.asarray(ef.pose.translation)
    f_quat_w = np.asarray(ef.pose.rotation.as_quat())
    t0_xyz, t0_quat = h_xyz_w[-1].copy(), h_quat_w[-1].copy()

    h_xyz, h_yaw = to_local(h_xyz_w, h_quat_w, t0_xyz, t0_quat)
    f_xyz, f_yaw = to_local(f_xyz_w, f_quat_w, t0_xyz, t0_quat)

    # vel/acc straight from the official interpolator (body frame), [16,2]
    h_vel = np.stack([np.asarray(eh.velocity[:, 0]), np.asarray(eh.velocity[:, 1])], axis=1)
    h_acc = np.stack([np.asarray(eh.acceleration[:, 0]), np.asarray(eh.acceleration[:, 1])], axis=1)

    hist = np.stack([h_xyz[:, 0], h_xyz[:, 1], h_yaw], axis=1)
    fut = np.stack([f_xyz[:, 0], f_xyz[:, 1], f_yaw], axis=1)

    nav = derive_nav(f_yaw)
    driving = [0.0] * 4
    driving[0 if nav == 1 else (2 if nav == 2 else 1)] = 1.0

    content, stds = [], []
    for view in VIEW_TAGS:
        content.append({"text": view})
        cam = CAMS_BY_VIEW[view]
        reader = avdi.get_clip_feature(clip, getattr(avdi.features.CAMERA, cam))
        for k, off_ms in enumerate(HIST_TS_OFFSETS):
            ts = np.array([t0 + off_ms * 1000], dtype=np.int64)
            frames, _ = reader.decode_images_from_timestamps(ts)
            img = frames[0]
            stds.append(float(np.asarray(img).std()))
            cur = (k == len(HIST_TS_OFFSETS) - 1)
            w, h = (CUR_W, CUR_H) if cur else (HIS_W, HIS_H)
            fname = f"{sid:04d}_{cam}_{k}.jpg"
            Image.fromarray(img).save(frames_dir / fname, quality=95)
            rel = f"frames/{fname}"
            content.append({"text": f"frame: {k}"})
            content.append({"image": rel, "resized_width": w, "resized_height": h})
    content.append({"text": instruction_template(hist, nav)})

    record = {
        "type": "chatml",
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": []},
        ],
        "meta_info": {"dataset": "physical_ai_chunk0_94", "token": f"chunk0_{sid:04d}",
                      "scene_token": str(clip), "cam_order": VIEW_TAGS},
        "trajectory": {
            "hist_traj_10hz": hist.round(6).tolist(),
            "hist_vel_10hz": h_vel.round(6).tolist(),
            "hist_acc_10hz": h_acc.round(6).tolist(),
            "future_traj_10hz": fut.round(6).tolist(),
            "future_valid_mask_10hz": [1] * N_FUT,
            "ego_status": {
                "ego_velocity": [round(float(h_vel[-1, 0]), 6), round(float(h_vel[-1, 1]), 6)],
                "ego_acceleration": [round(float(h_acc[-1, 0]), 6), round(float(h_acc[-1, 1]), 6)],
                "driving_command": driving,
            },
            "nav_command": nav,
        },
    }
    return record, min(stds)


def main():
    sys.path.insert(0, os.environ["QD_ALPAMAYO_RECIPES"] + "/src")
    from alpamayo.data.pai_utils import PhysicalAIAVDatasetLocalInterface

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-scenes", type=int, default=94)
    ap.add_argument("--pack", action="store_true", help="also build packed parquet")
    args = ap.parse_args()

    out = Path(args.out)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    avdi = PhysicalAIAVDatasetLocalInterface(local_dir=TRAIN_DIR, chunk_ids=[0])
    idx = avdi.clip_index.reset_index()
    clips = sorted(idx[(idx["chunk"] == 0) & (idx["clip_is_valid"])]["clip_id"].tolist())[: args.num_scenes]
    print(f"scenes: {len(clips)} (chunk 0)", flush=True)

    records, bad = [], []
    for si, clip in enumerate(clips):
        rec, min_std = encode_clip(avdi, clip, si, frames_dir)
        records.append(rec)
        if min_std < 15.0:
            bad.append(si)
        if si % 10 == 0:
            print(f"[{si}] {str(clip)[:10]} nav={rec['trajectory']['nav_command']}", flush=True)

    jsonl = out / "qwen_scenes_chunk0.jsonl"
    with open(jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {jsonl} ({len(records)} scenes); degenerate(min std<15): {bad if bad else 'none'}")

    if args.pack:
        import subprocess
        packer = Path(os.environ["QD_UPSTREAM_REPO"]) / "scripts" / "pack_images.py"
        cmd = [sys.executable, packer, "--scenes", str(jsonl),
               "--image-root", str(out), "--output", str(out / "archives" / "chunk0")]
        print("packing:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
