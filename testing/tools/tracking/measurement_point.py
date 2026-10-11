"""Measurement point study (step 6.3): nearest point, corrected centre or centroid as the position the tracker reads.

    python -m testing.tools.tracking.measurement_point --recording dev_default_seed12345
    python -m testing.tools.tracking.measurement_point --recording dev_default_seed12345 --ground height --camera
    python -m testing.tools.tracking.measurement_point --recording dev_default_seed12345 --sigma-a 1.0 --scans 0:150:1

--recording   folder name inside testing/data/recordings, or a path (needs the ground-truth files)
--ground      range_image (default) or height
--camera      use the camera edge refinement
--scans A:B:S scans to use (default all)
--sigma-a     acceleration noise of the filter, m/s^2 (default 1.5)
--burn-in     measurements of a hazard not counted at its start (default 10)
--min-samples hazards with fewer counted measurements are left out (default 15)
--out         output folder, default testing/data/measurement_point/<recording>

Writes summary.txt (printed too), spikes.txt (the measurements behind the big velocity errors), summary.csv, aggregate.csv, series.csv and a plot per hazard (velocity of the three
points against the true velocity). The columns are explained at the top of testing/support/measurement_point_study.py.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from server.perception.ground import GroundConfig, GroundMethod  # noqa: E402
from server.perception.pipeline import PerceptionConfig  # noqa: E402
from server.perception.camera_refinement import RefinementConfig  # noqa: E402
from testing.support.measurement_point_study import StudyConfig, aggregate, collect, spikes, study  # noqa: E402
from testing.support.paths import DATA_DIR  # noqa: E402
from testing.support.replay_scans import resolve  # noqa: E402


def plot_hazard(series, hid: int, label: str, path: Path):
    g = series[series["id"] == hid]
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for ax, comp in zip(axes[:2], ("vx", "vy")):
        for point, s in g.groupby("point"):
            ax.plot(s["t"], s[comp], label=point, lw=1)
        first = g[g["point"] == "nearest"]
        ax.plot(first["t"], first["true_" + comp], "k--", lw=1, label="true")
        ax.set_ylabel(comp + " (m/s)")
    for point, s in g.groupby("point"):
        axes[2].plot(s["t"], np.hypot(s["vx"], s["vy"]), label=point, lw=1)
    first = g[g["point"] == "nearest"]
    axes[2].plot(first["t"], np.hypot(first["true_vx"], first["true_vy"]), "k--", lw=1, label="true")
    axes[2].set_ylabel("speed (m/s)")
    axes[2].set_xlabel("t (s)")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].set_title(f"hazard {hid} ({label})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--ground", default="range_image", choices=[m.value for m in GroundMethod])
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--scans", default=None)
    ap.add_argument("--sigma-a", type=float, default=1.5)
    ap.add_argument("--burn-in", type=int, default=10)
    ap.add_argument("--min-samples", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    folder = resolve(args.recording)
    scans = None
    if args.scans:
        a, b, s = (int(v) for v in args.scans.split(":"))
        scans = range(a, b, s)
    config = PerceptionConfig(ground=GroundConfig(method=GroundMethod(args.ground)))
    samples = collect(folder, config, scans, use_camera=args.camera)
    cfg = StudyConfig(sigma_a=args.sigma_a, burn_in=args.burn_in, min_samples=args.min_samples)
    series, summary = study(samples, cfg=cfg)
    agg = aggregate(summary)
    text = (f"recording {folder.name}, ground {args.ground}{' + camera' if args.camera else ''}, sigma_a {args.sigma_a}, "
            f"burn-in {args.burn_in}, {len(samples)} hazard measurements\n\n"
            "per hazard and point (m/s unless noted; phantom_* for hazards that stand still)\n"
            + (summary.round(3).to_string(index=False) if len(summary) else "no hazard with enough measurements")
            + "\n\nper point (mean over the hazards that stand still / that move)\n"
            + (agg.round(3).to_string(index=False) if len(agg) else "-") + "\n")
    print(text)
    out = Path(args.out) if args.out else DATA_DIR / "measurement_point" / folder.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text(text)
    sp = spikes(series) if len(series) else None
    if sp is not None:
        (out / "spikes.txt").write_text("centre point: measurements with a velocity error above 1.5 m/s and the last 4 of each hazard\n" + sp.to_string(index=False) + "\n")
    summary.to_csv(out / "summary.csv", index=False)
    agg.to_csv(out / "aggregate.csv", index=False)
    series.to_csv(out / "series.csv", index=False)
    if len(summary):
        for hid, g in summary.groupby("id"):
            plot_hazard(series, int(hid), str(g["label"].iloc[0]), out / f"hazard_{int(hid)}.png")
    print("wrote", out)
    return summary


if __name__ == "__main__":
    main()