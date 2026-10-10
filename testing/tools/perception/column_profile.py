"""Print the depth and height profile of one cluster, column by column, with the split decision at every step.

    python -m testing.tools.perception.column_profile --recording dev_default_seed12345 --frame 115
    python -m testing.tools.perception.column_profile --recording dev_default_seed12345 --frame 115 --cluster 0

The cluster is taken before splitting (what the angle test alone joins); by default the one with most cells, or the one
with the given label (labels as in the picture of render_scan with splitting off). Written to
testing/data/profiles/<recording>_scanNNNNN.txt and printed. Columns:
  col       column of the range image            n    cells of the column in the cluster
  depth     smallest range of the column (m)      top  highest point of the column above the sensor frame floor (m)
  d_depth   jump to the next column                d_top  the same for the top
  cut       DEPTH / HEIGHT when the split test cuts there; the thresholds are printed in the header.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from server.geometry.calibration import Calibration
from server.perception.ground import GroundConfig, GroundMethod
from server.perception.pipeline import PerceptionConfig, PerceptionPipeline
from server.perception.segmentation import Segmenter
from testing.support.dev_recording import load_lidar
from testing.support.paths import DATA_DIR
from testing.support.replay_scans import lidar_message, resolve


def profile_text(pipe, img, result, label=None) -> str:
    seg_cfg = replace(pipe.config.segmentation, split=False, merge_caps=False)
    seg = Segmenter(pipe.builder.geometry, seg_cfg)
    raw = seg.segment(img, result.candidates)
    if not raw.clusters:
        return "no clusters in this scan\n"
    c = raw.clusters[label] if label is not None else max(raw.clusters, key=lambda k: len(k.rows))
    rng = np.where(result.candidates, img.range_m, np.nan)
    z = pipe.builder.geometry.points_vehicle(img)[..., 2]
    rows, cols = c.rows, c.cols
    ucols = np.unique(cols)
    pos = np.searchsorted(ucols, cols)
    depth = np.full(len(ucols), np.inf)
    top = np.full(len(ucols), -np.inf)
    n = np.bincount(pos)
    np.minimum.at(depth, pos, rng[rows, cols])
    np.maximum.at(top, pos, z[rows, cols])
    adjacent = np.diff(ucols) == 1
    cfg = seg.config
    vcell = float(np.median(rng[rows, cols])) * float(seg.row_step[max(rows.min() - 1, 0):rows.max() + 1].max())
    dd, dz = np.abs(np.diff(depth)), np.abs(np.diff(top))
    min_h = max(cfg.split_height_step_m, cfg.split_height_beam_factor * vcell)
    cut_d = adjacent & seg._is_step(dd, adjacent, cfg.split_depth_step_m)
    cut_h = adjacent & seg._is_step(dz, adjacent, min_h)
    out = [f"cluster {c.label}: {len(rows)} cells, columns {ucols.min()}..{ucols.max()}, median range {np.median(rng[rows, cols]):.1f} m, "
           f"rows {rows.min()}..{rows.max()}",
           f"largest beam gap here {vcell:.2f} m; depth step needs > max({cfg.split_depth_step_m}, 2.5 x neighbours); "
           f"height step needs > max({min_h:.2f}, 2.5 x neighbours)",
           f"{'col':>5} {'n':>4} {'depth':>8} {'top':>7} {'d_depth':>8} {'d_top':>7}  cut"]
    for i, u in enumerate(ucols):
        j = f"{dd[i]:8.2f} {dz[i]:7.2f}" if i < len(dd) and adjacent[i] else f"{'':>8} {'':>7}"
        mark = ("DEPTH " if i < len(cut_d) and cut_d[i] else "") + ("HEIGHT" if i < len(cut_h) and cut_h[i] else "")
        out.append(f"{u:5d} {n[i]:4d} {depth[i]:8.2f} {top[i]:7.2f} {j}  {mark}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--cluster", type=int, default=None)
    ap.add_argument("--ground", default="range_image", choices=[m.value for m in GroundMethod])
    ap.add_argument("--out", default=str(DATA_DIR / "profiles"))
    args = ap.parse_args()
    folder = resolve(args.recording)
    calib = Calibration.from_json(folder / "calibration.json")
    pipe = PerceptionPipeline(calib, PerceptionConfig(ground=GroundConfig(method=GroundMethod(args.ground))))
    rec = load_lidar(folder)[args.frame]
    img = pipe.builder.ingest(lidar_message(rec))
    result = pipe.process(img)
    text = profile_text(pipe, img, result, args.cluster)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{folder.name}_scan{args.frame:05d}.txt"
    path.write_text(text)
    print(text)
    print("wrote", path)


if __name__ == "__main__":
    main()