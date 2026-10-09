"""Summary of a development recording: python experiments/phase3/inspect_recording.py <recording folder>"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from testing.support.dev_recording import camera_path, load_camera_index, load_lidar, load_meta, load_pose  # noqa: E402


def intervals(t):
    iv = np.round(np.diff(np.asarray(t)) * 1000).astype(int)[1:]   # the first interval depends on the first step
    vals, cnt = np.unique(iv, return_counts=True)
    return ", ".join(f"{v} ms: {c} ({100 * c / len(iv):.1f}%)" for v, c in zip(vals, cnt))


folder = Path(sys.argv[1])
meta = load_meta(folder)
print(f"recording {folder.name}: scene {meta['scene']}, sensors '{meta['sensorConfig']}', seed {meta['seed']}, "
      f"Unity {meta['unityVersion']}, step {meta['fixedDeltaTime']} s")

lid = load_lidar(folder)
r = lid["ranges"]
print(f"\nLiDAR: {len(lid)} scans, {meta['rows']} x {meta['cols']}, t {lid['t'][0]:.3f} to {lid['t'][-1]:.3f} s")
print(f"  intervals: {intervals(lid['t'])}")
print(f"  mean fractions per scan: valid {np.mean((r > 0) & (r < 65535)):.3f}, no return {np.mean(r == 0):.4f}, max range or sky {np.mean(r == 65535):.3f}")
print(f"  sha256 of lidar.bin: {hashlib.sha256((folder / 'lidar.bin').read_bytes()).hexdigest()[:16]}...")

cam = load_camera_index(folder)
missing = sum(1 for f in cam["frame"] if not camera_path(folder, int(f)).exists())
print(f"\ncamera: {len(cam)} frames, intervals: {intervals(cam['t'])}")
print(f"  JPEG size mean {cam['bytes'].mean() / 1024:.1f} KB, max {cam['bytes'].max() / 1024:.1f} KB, files missing: {missing}")

pose = load_pose(folder)
print(f"\npose: {len(pose)} rows, intervals: {intervals(pose['t'])}")
print(f"  final reported pose: x {pose.x.iloc[-1]:.2f}, y {pose.y.iloc[-1]:.2f}, yaw {np.degrees(pose.yaw.iloc[-1]):.2f} deg")

if (folder / "ego.csv").exists():
    ego = pd.read_csv(folder / "ego.csv")
    haz = pd.read_csv(folder / "hazards.csv")
    info = pd.read_csv(folder / "hazard_info.csv")
    print(f"\nground truth: ego rows {len(ego)}, step {np.diff(ego.t).mean():.5f} s, final x {ego.x.iloc[-1]:.2f}, "
          f"peak speed {ego.speed.max():.2f} m/s, final speed {ego.speed.iloc[-1]:.2f}")
    for _, row in info.iterrows():
        g = haz[haz.id == row["id"]]
        v = g.iloc[1:]
        moving = v[v.speed > 0.01]
        vel = f"({moving.vx.median():.2f}, {moving.vy.median():.2f}) while moving" if len(moving) else "stationary"
        print(f"  #{row['id']} {row['label']}: start ({g.x.iloc[0]:.2f}, {g.y.iloc[0]:.2f}, {g.z.iloc[0]:.2f}), "
              f"yaw {np.degrees(g.yaw_rad.iloc[0]):.0f} deg, velocity {vel}")

    def hazard(label):
        sel = info[info.label == label]
        return haz[haz.id == sel.iloc[0]["id"]] if len(sel) else None

    cv = hazard("crossing_vehicle")
    if cv is not None:
        reached = cv[cv.y >= 0]
        if len(reached):
            c = reached.iloc[0]
            print(f"  crossing vehicle reaches y = 0 at t = {c.t:.2f} s, when the ego rear axle is at x = {np.interp(c.t, ego.t, ego.x):.1f} m")
        else:
            print("  crossing vehicle never reaches y = 0 in this recording")
    ped = hazard("pedestrian")
    if ped is not None:
        px = np.interp(ego.t, ped.t, ped.x)
        passed = ego.x.values >= px
        if passed.any():
            i = int(np.argmax(passed))
            print(f"  ego rear axle passes the walking pedestrian at t = {ego.t.iloc[i]:.2f} s (x = {ego.x.iloc[i]:.1f} m)")
        else:
            print("  ego never passes the walking pedestrian in this recording")