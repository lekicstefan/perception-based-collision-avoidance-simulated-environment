"""Offline evaluation of the perception on a development recording against the ground truth (step 5.7).

  python -m testing.tools.perception.evaluate --recording dev_default_seed12345
  python -m testing.tools.perception.evaluate --recording dev_default_seed12345 --compare
  python -m testing.tools.perception.evaluate --recording dev_default_seed12345 --ground height --camera
  python -m testing.tools.perception.evaluate --recording dev_default_seed12345 --scans 0:100:2 --min-hits 3

--recording   folder name inside testing/data/recordings, or a path (needs the ground-truth files of GroundTruthLogger)
--ground      range_image (default) or height
--camera      apply the camera edge refinement (needs the camera frames of the recording)
--compare     run all four combinations: both ground methods, with and without the camera
--min-hits    a hazard counts for recall when at least this many LiDAR cells hit it and returned (default 2)
--scans A:B:S scans to use (default all)
--out         output folder, default testing/data/evaluation/<recording>

Writes per configuration: hazards.csv (a row per hazard and scan), clusters.csv, summary.json, problems.txt (missed,
fragmented and outlier rows); and for the run:
summary.txt and the plots recall_vs_range.png, recall_vs_hits.png, errors_vs_range.png, timeline.png.
Recall and the errors are explained at the top of testing/support/perception_eval.py.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from server.perception.camera_refinement import RefinementConfig  # noqa: E402
from server.perception.ground import GroundConfig, GroundMethod  # noqa: E402
from server.perception.pipeline import PerceptionConfig  # noqa: E402
from testing.support.evaluation_run import evaluate_recording  # noqa: E402
from testing.support.paths import DATA_DIR  # noqa: E402
from testing.support.perception_eval import EvalConfig, RANGE_BINS, format_summary, summarise  # noqa: E402
from testing.support.replay_scans import resolve  # noqa: E402


def config_name(ground: str, camera: bool) -> str:
    return ground + ("+camera" if camera else "")


def problems_text(haz) -> str:
    """The rows behind the bad numbers: missed hazards, hazards split into several clusters, and outliers."""
    cols = ["scan", "t", "label", "true_range_m", "hits", "n_clusters", "cluster_sizes", "in_clusters", "own_pts", "inside_pts", "stolen_pts", "owners",
            "other_id", "gap_other_m", "step_other_m", "nearest_err", "width_err"]
    det = haz[haz["detectable"]]
    parts = []
    for title, rows in (("MISSED (detectable, no cluster)", det[~det["detected"]]),
                        ("FRAGMENTED (more than one cluster)", det[det["detected"] & (det["n_clusters"] > 1)]),
                        ("OUTLIERS (|nearest_err| > 0.3 m or |width_err| > 1 m)",
                         det[det["detected"] & ((det["nearest_err"].abs() > 0.3) | (det["width_err"].abs() > 1.0))])):
        parts.append(f"{title}: {len(rows)} rows" + ("" if rows.empty else "\n" + rows[cols].round(2).to_string(index=False)))
    return "\n\n".join(parts) + "\n"


def plot_recall(results: dict, key: str, xlabel: str, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, (summary, _, _) in results.items():
        data = summary.get(key, {})
        if data:
            ax.plot(list(data.keys()), [v[0] for v in data.values()], marker="o", label=name)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("recall")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_errors(results: dict, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    titles = [("centre_err", "corrected centre error (m)"), ("centroid_err", "centroid error (m)"),
              ("nearest_err", "nearest-point range error (m)")]
    for ax, (col, title) in zip(axes, titles):
        for name, (_, haz, _) in results.items():
            d = haz[haz["detectable"] & haz["detected"]]
            if col in d:
                ax.scatter(d["true_range_m"], d[col], s=8, alpha=0.5, label=name)
        ax.set_title(title)
        ax.set_xlabel("true range (m)")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_timeline(results: dict, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, (_, _, cl) in results.items():
        if len(cl):
            per = cl.groupby("scan").agg(clusters=("label", "size"), matched=("matched", "sum"))
            ax.plot(per.index, per["clusters"], label=f"{name}: clusters")
            ax.plot(per.index, per["matched"], linestyle="--", label=f"{name}: matched to a hazard")
    ax.set_xlabel("scan")
    ax.set_ylabel("clusters")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--ground", default="range_image", choices=[m.value for m in GroundMethod])
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--min-hits", type=int, default=2)
    ap.add_argument("--scans", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    folder = resolve(args.recording)
    out = Path(args.out) if args.out else DATA_DIR / "evaluation" / folder.name
    out.mkdir(parents=True, exist_ok=True)
    scans = None
    if args.scans:
        a, b, s = (int(x) for x in args.scans.split(":"))
        scans = range(a, b, s)
    combos = ([(g.value, c) for g in GroundMethod for c in (False, True)] if args.compare else [(args.ground, args.camera)])

    results, text = {}, []
    for ground, camera in combos:
        name = config_name(ground, camera)
        cfg = PerceptionConfig(ground=GroundConfig(method=GroundMethod(ground)), refinement=RefinementConfig(enabled=camera))
        haz, cl = evaluate_recording(folder, cfg, EvalConfig(min_hits=args.min_hits), scans=scans, use_camera=camera)
        summary = summarise(haz, cl)
        sub = out / name
        sub.mkdir(exist_ok=True)
        haz.to_csv(sub / "hazards.csv", index=False)
        cl.to_csv(sub / "clusters.csv", index=False)
        (sub / "problems.txt").write_text(problems_text(haz), encoding="utf-8")
        (sub / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        results[name] = (summary, haz, cl)
        text.append(format_summary(name, summary))
        print(text[-1], flush=True)
    (out / "summary.txt").write_text("\n\n".join(text) + "\n", encoding="utf-8")
    plot_recall(results, "recall_by_range", "true range to the hazard (m)", out / "recall_vs_range.png")
    plot_recall(results, "recall_by_hits", "LiDAR cells that hit the hazard and returned", out / "recall_vs_hits.png")
    plot_errors(results, out / "errors_vs_range.png")
    plot_timeline(results, out / "timeline.png")
    print("wrote", out)
    return results


if __name__ == "__main__":
    main()