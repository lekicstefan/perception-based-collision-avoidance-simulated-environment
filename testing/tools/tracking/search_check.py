"""Long parameter search for the tracker (step 6.9): random search + refinement against the ground truth, LiDAR only.

    python -m testing.tools.tracking.search_check --recording dev_default_seed12345
    python -m testing.tools.tracking.search_check --recording rec_a,rec_b --holdout rec_c --random 600 --rounds 6 --workers 6

--recording   comma list of recordings (names in testing/data/recordings or paths) the score is computed on (mean over them)
--holdout     comma list of recordings NOT used in the search; the default and the best set are compared on them at the end
--random      random parameter sets (default 300);   --rounds refinement rounds (default 5);   --per-round perturbations of each
              of the --top best (default 8 and 5): evaluations = random + rounds * top * per-round
--workers     processes (default: CPU count - 1; 1 = no multiprocessing)
--ground      range_image (default) or height;   --scans A:B:S;   --seed 0;   --burn-in 10
--out         output folder, default testing/data/search/<first recording>

Before the search it times one evaluation and prints the estimated total time. Parameters searched: sigma_a, sigma_extent
(centre noise), gate, known_sigma, static_max_speed, static_confirm_scans, leave_chi2. Score: top of
testing/support/search_study.py (lower is better; a mover flagged static counts most). Writes summary.txt (default vs best,
top sets, sensitivity, holdout) and results.csv (every evaluated set).

Be careful with the result: a search on one recording with five hazards can tune to that recording. Use --holdout with a recording
of another seed or scenario, and prefer parameters that stay good there and in the sensitivity table lie in a wide range.
"""
import argparse
import os
import time
import warnings
from pathlib import Path

import pandas as pd

from server.perception.ground import GroundConfig, GroundMethod
from server.perception.pipeline import PerceptionConfig
from testing.support.association_study import scan_measurements
from testing.support.paths import DATA_DIR
from testing.support.replay_scans import resolve
from testing.support.search_study import DEFAULT, SPACE, Evaluator, run_params, score, search, sensitivity
from testing.support.tuning_study import metrics


def _load(names: str, ground: str, scans):
    cfg = PerceptionConfig(ground=GroundConfig(method=GroundMethod(ground)))
    out = []
    for n in [v.strip() for v in names.split(",") if v.strip()]:
        folder = resolve(n)
        out.append((folder, scan_measurements(folder, cfg, scans, use_camera=True)))
    return out


def _holdout_table(holdout: list, default: dict, best: dict, burn_in: int) -> pd.DataFrame:
    rows = []
    for folder, scans in holdout:
        md, mb = (metrics(run_params(folder, scans, p, burn_in)) for p in (default, best))
        ref = md["rows"]
        for name, m in (("default", md), ("best", mb)):
            rows.append(dict(recording=folder.name, config=name, score=score(m, ref), **{k: v for k, v in m.items() if k != "nis"}))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--holdout", default="")
    ap.add_argument("--random", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--per-round", type=int, default=8)
    ap.add_argument("--top", type=int, default=5)
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
    t0 = time.time()
    ev = Evaluator(context, args.workers, args.burn_in)
    one = (time.time() - t0) / max(len(context), 1)
    print(f"{len(context)} recording(s), {n_eval} evaluations, {one:.1f} s for one (serial); about "
          f"{one * n_eval / max(args.workers, 1) / 60:.0f} min with {args.workers} worker(s)")
    try:
        table = search(ev, args.random, args.rounds, args.per_round, args.top, args.seed)
    finally:
        ev.close()
    default = table[(table[list(SPACE)] == pd.Series(DEFAULT)).all(axis=1)].head(1)
    best = {k: table.iloc[0][k] for k in SPACE}
    best = {k: (int(v) if SPACE[k][2] else float(v)) for k, v in best.items()}
    cols = list(SPACE) + ["score", "worst_recording_score", "rows", "vel_rms_moving", "vel_rms_still", "static_while_moving_max",
                          "static_while_still_mean", "leave_latency_max_s", "pos_rms"]
    text = (f"recordings {[f.name for f, _ in context]}, {len(table)} evaluations, workers {args.workers}, "
            f"{(time.time() - t0) / 60:.1f} min\n\ndefault configuration:\n" + default[cols].round(3).to_string(index=False)
            + "\n\nbest 10:\n" + table[cols].head(10).round(3).to_string(index=False)
            + f"\n\nbest configuration (to put into the defaults):\n{best}\n\nsensitivity (the best 10 % of the sets; "
              "range_used near 1 = the parameter hardly matters, near 0 = it is pinned down):\n"
            + sensitivity(table).round(3).to_string(index=False) + "\n")
    if holdout:
        text += "\nholdout recordings (not used in the search; score with the default's rows as reference):\n" + \
                _holdout_table(holdout, DEFAULT, best, args.burn_in).round(3).to_string(index=False) + "\n"
    print(text)
    out = Path(args.out) if args.out else DATA_DIR / "search" / context[0][0].name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text(text)
    table.to_csv(out / "results.csv", index=False)
    print("wrote", out)
    return table


if __name__ == "__main__":
    main()