"""Tune the tracker against the ground truth (step 6.9): one parameter at a time around the defaults.

    python -m testing.tools.tracking.tuning_check --recording dev_default_seed12345
    python -m testing.tools.tracking.tuning_check --recording dev_default_seed12345 --sigma-px 0.5,1,2 --camera-every 3

--recording     folder name inside testing/data/recordings, or a path (needs ground-truth files)
--ground        range_image (default) or height;   --scans A:B:S
--sigma-a       comma list, default 0.5,1,1.5,2.5,4   (acceleration noise, m/s^2)
--sigma-extent  comma list, default 0.3,0.5,0.7,1,1.5 (centre noise along the line of sight, m)
--gate          comma list, default 5.99,9.21,13.8,18.4 (association chi-square, 2 dof)
--sigma-px      comma list, default none: camera bearing noise in pixels; runs with the camera (slow), also gives the NIS
--sigma-extra   comma list, default none: pose/calibration part of the bearing sigma in rad (default of the tracker 0.002); camera run
--camera-every  use every n-th camera frame in the --sigma-px runs (default 3)
--burn-in       confirmed measurements of a hazard not counted at its start (default 10)
--out           output folder, default testing/data/tuning/<recording>

Columns: rows (scored measurements; fewer means lost tracks), vel_rms_moving / vel_rms_still (velocity error of movers and of
hazards that stand still), phantom_rms (speed of still hazards), static_while_moving_max (the dangerous one: a mover flagged
static, worst hazard), static_while_still_mean, leave_latency_max_s, pos_rms, nis (camera runs, about 1 when sigma is right).
Writes summary.txt and summary.csv.
"""
import argparse
from pathlib import Path

from server.perception.ground import GroundConfig, GroundMethod
from server.perception.pipeline import PerceptionConfig
from testing.support.association_study import scan_measurements
from testing.support.paths import DATA_DIR
from testing.support.replay_scans import resolve
from testing.support.tuning_study import sweep

DEFAULT_GRID = dict(sigma_a="0.5,1,1.5,2.5,4", sigma_extent="0.3,0.5,0.7,1,1.5", gate="5.99,9.21,13.8,18.4", sigma_px="", sigma_extra="")


def _floats(s: str) -> list:
    return [float(v) for v in s.split(",") if v.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--ground", default="range_image", choices=[m.value for m in GroundMethod])
    ap.add_argument("--scans", default=None)
    for k, v in DEFAULT_GRID.items():
        ap.add_argument("--" + k.replace("_", "-"), default=v)
    ap.add_argument("--camera-every", type=int, default=3)
    ap.add_argument("--burn-in", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    folder = resolve(args.recording)
    scans = None
    if args.scans:
        a, b, s = (int(v) for v in args.scans.split(":"))
        scans = range(a, b, s)
    grid = {k: _floats(getattr(args, k)) for k in DEFAULT_GRID}
    grid = {k: v for k, v in grid.items() if v}
    data = scan_measurements(folder, PerceptionConfig(ground=GroundConfig(method=GroundMethod(args.ground))), scans, use_camera=True)
    table = sweep(folder, data, grid, camera_every=args.camera_every, burn_in=args.burn_in)
    text = (f"recording {folder.name}, {len(data)} scans; one parameter at a time, the others at their defaults "
            f"(sigma_a 1.4, sigma_extent 0.4, gate 18.4, the tuned values of 6.9)\n\n" + table.round(3).to_string(index=False) + "\n")
    print(text)
    out = Path(args.out) if args.out else DATA_DIR / "tuning" / folder.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text(text)
    table.to_csv(out / "summary.csv", index=False)
    print("wrote", out)
    return table


if __name__ == "__main__":
    main()