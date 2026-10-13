"""LiDAR + camera fusion check (step 6.6): the tracker with and without the camera bearing updates, against the ground truth.

    python -m testing.tools.tracking.fusion_check --recording dev_default_seed12345
    python -m testing.tools.tracking.fusion_check --recording dev_default_seed12345 --camera-every 2 --sigma-px 3

--recording     folder name inside testing/data/recordings, or a path (needs ground-truth files and camera frames)
--ground        range_image (default) or height;   --scans A:B:S
--camera-every  use every n-th camera frame (default 1: all)
--sigma-px      pixel standard deviation of a bearing (default 2);   --gate  bearing gate, chi-square 1 dof (default 9)
--sigma-a       acceleration noise of the filter (default 1.4)
--margin-frac   ROI margin as a fraction of the expected width (default 0.48)
--min-edge-frac a column is an edge when edge pixels fill this fraction of the rows (default 0.1)
--canny-low/--canny-high  Canny thresholds (default 41 / 188);   --ambiguity  second-edge ratio that makes a side ambiguous (default 0.68)
--no-color      edges of the gray image only (default: also of each colour channel)
--no-extent     without the size memory of the tracks (step 6.11), for the comparison; use --out to keep both results
--burn-in       confirmed measurements of a hazard not counted at its start (default 10)
--out           output folder, default testing/data/fusion/<recording>

Prints and writes summary.txt: per hazard the position error (also across and along the line of sight), the velocity error and
the phantom speed of static hazards, without and with the camera; the static flag against the true motion (also while moving
and how long it takes to clear when a hazard starts to move); and the filter against raw two-frame differencing; then how many camera updates were applied and the reasons
for the others. Columns: top of testing/support/fusion_study.py.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from server.perception.ground import GroundConfig, GroundMethod
from server.perception.pipeline import PerceptionConfig
from server.tracking.bearing import BearingConfig
from server.tracking.extent import ExtentConfig
from testing.support.association_study import scan_measurements
from testing.support.fusion_study import replay_fusion, summarise_ablation, summarise_fusion, summarise_outputs
from testing.support.paths import DATA_DIR
from testing.support.replay_scans import resolve


def bearing_config(args) -> BearingConfig:
    edge = {k: v for k, v in dict(canny_low=args.canny_low, canny_high=args.canny_high, min_edge_frac=args.min_edge_frac,
                                   ambiguity_ratio=args.ambiguity).items() if v is not None}
    top = {} if args.margin_frac is None else dict(margin_frac=args.margin_frac)
    return BearingConfig(sigma_px=args.sigma_px, gate_chi2=args.gate, edges=replace(BearingConfig().edges, **edge), use_color=not args.no_color, **top)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--ground", default="range_image", choices=[m.value for m in GroundMethod])
    ap.add_argument("--scans", default=None)
    ap.add_argument("--camera-every", type=int, default=1)
    ap.add_argument("--sigma-px", type=float, default=2.0)
    ap.add_argument("--gate", type=float, default=9.0)
    ap.add_argument("--sigma-a", type=float, default=1.4)
    ap.add_argument("--margin-frac", type=float, default=None)
    ap.add_argument("--min-edge-frac", type=float, default=None)
    ap.add_argument("--canny-low", type=float, default=None)
    ap.add_argument("--canny-high", type=float, default=None)
    ap.add_argument("--ambiguity", type=float, default=None)
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--no-extent", action="store_true")
    ap.add_argument("--burn-in", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    folder = resolve(args.recording)
    scans = None
    if args.scans:
        a, b, s = (int(v) for v in args.scans.split(":"))
        scans = range(a, b, s)
    data = scan_measurements(folder, PerceptionConfig(ground=GroundConfig(method=GroundMethod(args.ground))), scans, use_camera=True)
    bcfg = bearing_config(args)
    runs = {name: replay_fusion(folder, data, use_cam, sigma_a=args.sigma_a, bearing=bcfg, camera_every=args.camera_every,
                              burn_in=args.burn_in, extent=ExtentConfig(enabled=not args.no_extent))
            for name, use_cam in (("lidar", False), ("lidar+camera", True))}
    tables = []
    for name, r in runs.items():
        s = summarise_fusion(r["rows"]) if len(r["rows"]) else pd.DataFrame()
        s.insert(0, "mode", name)
        tables.append(s)
    table = pd.concat(tables, ignore_index=True)
    if len(table):
        table = table.sort_values(["id", "mode"])
    cam = runs["lidar+camera"]
    motion = summarise_outputs(cam["rows"])
    ablation = summarise_ablation(cam["rows"])
    text = (f"recording {folder.name}, {len(data)} scans, camera frames used {cam['camera_frames']} (every {args.camera_every}), "
            f"sigma {args.sigma_px} px, gate {args.gate}, size memory {'off' if args.no_extent else 'on'}\n\n" + table.round(3).to_string(index=False)
            + "\n\nmotion flags and outputs (lidar+camera): static_rate should be ~1 for hazards that stand still and ~0 for movers;"
              " speed_out_rms is the speed after the static flag removed the phantom velocity\n" + motion.round(3).to_string(index=False)
            + "\n\nfilter against raw two-frame differencing (lidar+camera): raw_over_filter = how many times larger the raw velocity error is;"
              " raw_static_* = the naive flag 'raw speed under 0.5 m/s' (static_while_moving is the dangerous one)\n" + ablation.round(3).to_string(index=False)
            + f"\n\ncamera track-frames by result: {cam['reasons']}\nmean normalised innovation squared of the applied updates: "
              f"{cam['nis_mean']:.2f} (about 1 when sigma is right)\n")
    print(text)
    out = Path(args.out) if args.out else DATA_DIR / "fusion" / folder.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text(text)
    table.to_csv(out / "summary.csv", index=False)
    print("wrote", out)
    return table


if __name__ == "__main__":
    main()