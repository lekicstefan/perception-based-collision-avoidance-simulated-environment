"""Search of the camera edge parameters (step 6.9, camera part): random search + refinement against the true silhouettes.

    python -m testing.tools.tracking.camera_search_check --recording dev_default_seed12345 --holdout dev_lidar_clean_seed12345
    python -m testing.tools.tracking.camera_search_check --recording dev_default_seed12345 --random 300 --rounds 5 --workers 8

--recording     comma list of recordings (with camera frames and ground truth) the score is computed on (mean over them)
--holdout       comma list of recordings NOT used in the search; the default and the best set are compared on them at the end
--random        random parameter sets (default 120);  --rounds refinement rounds (default 4);  --per-round perturbations of each of
                the --top best (default 8 and 4): evaluations = 1 + random + rounds * top * per-round
--camera-every  use every n-th camera frame (default 3; 1 = all, three times slower)
--tol-px        both edges within this many pixels of the true silhouette count as good (default 5)
--workers       processes (default: CPU count - 1);  --ground;  --scans A:B:S;  --seed;  --burn-in
--out           output folder, default testing/data/camera_search/<first recording>

Searched: Canny thresholds (low, high/low), blur, edge-column fraction, minimum contrast, ambiguity ratio, ROI margins, minimum
and plausible width, colour channels on/off and their threshold factor. The score is at the top of
testing/support/camera_search_study.py (a false edge counts double, a mover flagged static counts ten times). Before the search one
evaluation is timed and the total time estimated. Writes summary.txt (default vs best, top sets, sensitivity, holdout) and results.csv.
A search on one recording can tune to it: look at the holdout and at the sensitivity table.
"""
import argparse
import os
import time
import warnings
from pathlib import Path

import pandas as pd

from server.perception.ground import GroundMethod
from testing.support.camera_search_study import DEFAULT, METRICS, SPACE, Evaluator, build_bearing, edge_metrics, score, search, sensitivity
from testing.support.paths import DATA_DIR
from testing.tools.tracking.search_check import _load


def _holdout_table(holdout: list, best: dict, camera_every: int, burn_in: int, tol_px: float) -> pd.DataFrame:
    rows = []
    for folder, scans in holdout:
        for name, p in (("default", DEFAULT), ("best", best)):
            m = edge_metrics(folder, scans, build_bearing(p), camera_every, burn_in, tol_px)
            rows.append(dict(recording=folder.name, config=name, score=score(m), **m))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--holdout", default="")
    ap.add_argument("--random", type=int, default=120)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--per-round", type=int, default=8)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--camera-every", type=int, default=3)
    ap.add_argument("--tol-px", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    ap.add_argument("--ground", default="range_image", choices=[m.value for m in GroundMethod])
    ap.add_argument("--scans", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--burn-in", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    scans = None
    if args.scans:
        a, b, s = (int(v) for v in args.scans.split(":"))
        scans = range(a, b, s)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    context = _load(args.recording, args.ground, scans)
    holdout = _load(args.holdout, args.ground, scans) if args.holdout else []
    n_eval = 1 + args.random + args.rounds * args.top * args.per_round
    ev = Evaluator(context, args.workers, args.camera_every, args.burn_in, args.tol_px)
    t0 = time.time()
    ev.evaluate([DEFAULT])
    one = time.time() - t0
    print(f"{len(context)} recording(s), {n_eval} evaluations, {one:.0f} s for one (serial); about "
          f"{one * n_eval / max(args.workers, 1) / 60:.0f} min with {args.workers} worker(s)")
    t0 = time.time()
    try:
        table = search(ev, args.random, args.rounds, args.per_round, args.top, args.seed)
    finally:
        ev.close()
    cols = list(SPACE) + ["score"] + METRICS
    best = {k: table.iloc[0][k] for k in SPACE}
    best = {k: (int(v) if SPACE[k][2] in ("i", "odd", "b") else float(v)) for k, v in best.items()}
    default = table[(table[list(SPACE)] == pd.Series(DEFAULT)).all(axis=1)].head(1)
    text = (f"recordings {[f.name for f, _ in context]}, {len(table)} evaluations, workers {args.workers}, camera every {args.camera_every}, "
            f"tolerance {args.tol_px} px, {(time.time() - t0) / 60:.1f} min\n\ndefault configuration:\n"
            + default[cols].round(3).to_string(index=False) + "\n\nbest 10:\n" + table[cols].head(10).round(3).to_string(index=False)
            + f"\n\nbest configuration:\n{best}\n\nsensitivity (the best 10 % of the sets; range_used near 1 = the parameter hardly matters, "
              "near 0 = it is pinned down):\n" + sensitivity(table).round(3).to_string(index=False) + "\n")
    if holdout:
        text += ("\nholdout recordings (not used in the search):\n"
                 + _holdout_table(holdout, best, args.camera_every, args.burn_in, args.tol_px).round(3).to_string(index=False) + "\n")
    print(text)
    out = Path(args.out) if args.out else DATA_DIR / "camera_search" / context[0][0].name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text(text)
    table.to_csv(out / "results.csv", index=False)
    print("wrote", out)
    return table


if __name__ == "__main__":
    main()