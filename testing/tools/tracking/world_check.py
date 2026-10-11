"""World-frame check of a development recording (step 6.1): do the clusters land where the hazards really are?

    python -m testing.tools.tracking.world_check --recording dev_default_seed12345
    python -m testing.tools.tracking.world_check --recording dev_default_seed12345 --ground height --scans 0:150:1

For every scan the clusters are moved into the world (start) frame with the pose stamped on the LiDAR message and matched to
the ground-truth hazards (the association of the evaluation, step 5.7). Printed and written to
testing/data/world_check/<recording>.txt:

  ego       yaw range of the run (the check of the yaw sign needs a run in which the car turns), distance driven
  pose      the pose in the LiDAR headers against the true ego pose of ego.csv (position and yaw, maximum and mean)
  hazards   per hazard, over the scans in which it was found: how many scans; the spread (standard deviation of x and y,
            in metres) of the world position of the nearest point, the corrected centre and the centroid; for a hazard that
            does not move this spread is the phantom movement the tracker would see, for a moving one the spread contains
            the real movement; and the mean world error of the centre against the true box centre.
A hazard that does not move and whose centre spread is far from the other hazards, or a pose difference that grows with the
time, points at a wrong frame, a wrong yaw sign or a wrong time stamp.
"""
import argparse
import math
from pathlib import Path

import numpy as np

from server.geometry.calibration import Calibration
from server.geometry.frames import wrap_angle
from server.perception.ground import GroundConfig, GroundMethod
from server.perception.pipeline import PerceptionConfig, PerceptionPipeline
from server.tracking.measurement import clusters_to_world
from testing.support.ground_truth import GroundTruth
from testing.support.paths import DATA_DIR
from testing.support.perception_eval import EvalConfig, associate
from testing.support.replay_scans import resolve, run_pipeline


def hazard_world(truth: GroundTruth, hid: int, t: float):
    g = truth.hazards[truth.hazards["id"] == hid].sort_values("t")
    return np.array([np.interp(t, g["t"], g["x"]), np.interp(t, g["t"], g["y"])]), float(g["speed"].max())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--ground", default="range_image", choices=[m.value for m in GroundMethod])
    ap.add_argument("--scans", default=None, help="A:B:S")
    ap.add_argument("--out", default=str(DATA_DIR / "world_check"))
    args = ap.parse_args()
    folder = resolve(args.recording)
    pipe = PerceptionPipeline(Calibration.from_json(folder / "calibration.json"),
                              PerceptionConfig(ground=GroundConfig(method=GroundMethod(args.ground))))
    truth = GroundTruth.load(folder)
    scans = None
    if args.scans:
        a, b, s = (int(v) for v in args.scans.split(":"))
        scans = range(a, b, s)
    sensor_xy = pipe.builder.geometry.origin[:2]
    cfg = EvalConfig()
    pose_err, found, labels, speeds = [], {}, {}, {}
    for k, rec, result in run_pipeline(folder, pipe, scans=scans):
        img = result.image
        ex, ey, eyaw = truth.ego_at(img.t)
        pose_err.append((math.hypot(img.pose[0] - ex, img.pose[1] - ey), abs(wrap_angle(img.pose[3] - eyaw)), img.pose[3]))
        objs = truth.objects_at(img.t)
        owner = associate(result.clusters, objs, cfg)
        world = clusters_to_world(result.clusters, img, sensor_xy)
        best = {}
        for w, o in zip(world, owner):
            if o is not None and (objs[o].id not in best or w.n_cells > best[objs[o].id].n_cells):
                best[objs[o].id] = w
        for o in objs:
            labels[o.id] = o.label
        for hid, w in best.items():
            true_xy, speeds[hid] = hazard_world(truth, hid, img.t)
            found.setdefault(hid, []).append((w.nearest_xy, w.centre_xy, w.centroid_xy, true_xy))
    pe = np.array(pose_err)
    yaws = np.degrees(pe[:, 2])
    lines = [f"recording {folder.name}, {len(pe)} scans, ground {args.ground}",
             f"ego: yaw from {yaws.min():+.1f} to {yaws.max():+.1f} deg" +
             ("" if np.ptp(yaws) > 5.0 else "   (the car does not turn: this recording cannot check the yaw sign)"),
             f"pose in the headers against ego.csv: position max {pe[:, 0].max():.3f} m, mean {pe[:, 0].mean():.3f} m; "
             f"yaw max {math.degrees(pe[:, 1].max()):.2f} deg, mean {math.degrees(pe[:, 1].mean()):.2f} deg", "",
             f"{'id':>3} {'label':>16} {'speed':>6} {'scans':>6} | spread std x/y (m): {'nearest':>13} {'centre':>13} {'centroid':>13}"
             f" | centre error (m): mean x, y, norm"]
    for hid, rows in sorted(found.items()):
        near, cen, cent, true = (np.array([r[i] for r in rows]) for i in range(4))
        err = (cen - true).mean(axis=0)
        sd = lambda a: f"{a[:, 0].std():5.2f}/{a[:, 1].std():5.2f}"  # noqa: E731
        lines.append(f"{hid:>3} {labels[hid]:>16} {speeds[hid]:6.1f} {len(rows):6d} | {'':23}{sd(near):>13} {sd(cen):>13} {sd(cent):>13}"
                     f" | {err[0]:+6.2f} {err[1]:+6.2f} {np.linalg.norm(err):5.2f}")
    text = "\n".join(lines) + "\n"
    print(text)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{folder.name}.txt").write_text(text)
    print("wrote", out / f"{folder.name}.txt")


if __name__ == "__main__":
    main()