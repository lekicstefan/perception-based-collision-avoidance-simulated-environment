import sys
from pathlib import Path

import numpy as np
import pandas as pd

folder = Path(sys.argv[1])
ego = pd.read_csv(folder / "ego.csv")
haz = pd.read_csv(folder / "hazards.csv")
vis = pd.read_csv(folder / "visibility.csv")
info = pd.read_csv(folder / "hazard_info.csv")

print(info.to_string(index=False))
dt = np.diff(ego["t"])
print(f"\nego rows {len(ego)}, step mean {dt.mean():.5f} s, min {dt.min():.5f}, max {dt.max():.5f}")
print(f"ego final: x {ego.x.iloc[-1]:.3f} y {ego.y.iloc[-1]:.3f} yaw {np.degrees(ego.yaw_rad.iloc[-1]):.2f} deg, speed {ego.speed.iloc[-1]:.2f}")

print("\nhazard trajectories (start frame):")
for hid, g in haz.groupby("id"):
    row = info[info.id == hid].iloc[0]
    v = g.iloc[1:]
    print(f"  #{hid} {row['name']} ({row.label}): rows {len(g)}, first pos ({g.x.iloc[0]:.2f}, {g.y.iloc[0]:.2f}, {g.z.iloc[0]:.2f}), "
          f"yaw {np.degrees(g.yaw_rad.iloc[0]):.1f} deg, velocity ({v.vx.mean():.3f}, {v.vy.mean():.3f}) m/s, "
          f"size {g.length.iloc[0]:.2f} x {g.width.iloc[0]:.2f} x {g.height.iloc[0]:.2f}")

print("\nvisibility per hazard (LiDAR scans):")
for hid, g in vis.groupby("id"):
    print(f"  #{hid}: scans {len(g)}, hits geometric median {g.hits_geometric.median():.0f} "
          f"(min {g.hits_geometric.min()}, max {g.hits_geometric.max()}), returned median {g.hits_returned.median():.0f}, "
          f"nearest range median {g.min_range_m.median():.2f} m, in view {100 * g.in_view.mean():.0f}% of scans")