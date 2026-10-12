"""Score named parameter sets on recordings, no search (step 6.9): compare the default, the search's best and your own candidates.

    python -m testing.tools.tracking.candidate_check --recording dev_default_seed12345,dev_lidar_clean_seed12345 ^
        --candidates "best:sigma_a=1.356,sigma_extent=0.32,gate=20,known_sigma=1.47,static_max_speed=0.73,static_confirm_scans=3,leave_chi2=10.2;safe:sigma_a=1.4,sigma_extent=0.4,gate=18.4,known_sigma=1.0,static_max_speed=0.6,static_confirm_scans=4"

--recording   comma list of recordings;   --candidates  'name:key=value,key=value;name2:...' (keys of SPACE, the others at the default;
              the default itself is always the first row);   --ground, --scans, --burn-in as in search_check
--out         output folder, default testing/data/candidates/<first recording>

Per recording and candidate: the score of search_study (reference rows = the default's), the metrics, and per hazard the
static_while_moving rate and the static_while_still rate (summary.txt).
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd

from testing.support.fusion_study import summarise_outputs
from testing.support.paths import DATA_DIR
from testing.support.search_study import DEFAULT, SPACE, clip, run_params, score
from testing.support.tuning_study import metrics
from testing.tools.tracking.search_check import _load


def parse_candidates(text: str) -> dict:
    out = {"default": dict(DEFAULT)}
    for part in [p.strip() for p in text.split(";") if p.strip()]:
        name, _, body = part.partition(":")
        p = dict(DEFAULT)
        for kv in [v for v in body.split(",") if v.strip()]:
            k, _, v = kv.partition("=")
            k = k.strip()
            if k not in SPACE:
                raise ValueError(f"unknown parameter {k!r}; choose from {list(SPACE)}")
            p[k] = float(v)
        out[name.strip()] = p
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--candidates", default="")
    ap.add_argument("--ground", default="range_image")
    ap.add_argument("--scans", default=None)
    ap.add_argument("--burn-in", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    scans = None
    if args.scans:
        a, b, s = (int(v) for v in args.scans.split(":"))
        scans = range(a, b, s)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cands = parse_candidates(args.candidates)
    context = _load(args.recording, args.ground, scans)
    rows, hazards = [], []
    for folder, data in context:
        ref = None
        for name, p in cands.items():
            p = {k: (v if k not in SPACE else clip({**DEFAULT, k: v})[k]) for k, v in p.items()}
            run = run_params(folder, data, p, args.burn_in)
            m = metrics(run)
            ref = m["rows"] if ref is None else ref
            rows.append(dict(recording=folder.name, candidate=name, score=score(m, ref), **{k: v for k, v in m.items() if k != "nis"}))
            h = summarise_outputs(run["rows"])
            h.insert(0, "candidate", name)
            h.insert(0, "recording", folder.name)
            hazards.append(h[["recording", "candidate", "id", "label", "true_speed", "static_rate", "static_while_moving",
                              "static_while_still", "leave_latency_s"]])
    table, haz = pd.DataFrame(rows), pd.concat(hazards, ignore_index=True)
    text = ("candidates:\n" + "\n".join(f"  {n}: {p}" for n, p in cands.items()) + "\n\n" + table.round(3).to_string(index=False)
            + "\n\nper hazard (static_while_moving must be ~0, static_while_still should be high):\n" + haz.round(3).to_string(index=False) + "\n")
    print(text)
    out = Path(args.out) if args.out else DATA_DIR / "candidates" / context[0][0].name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text(text)
    table.to_csv(out / "summary.csv", index=False)
    print("wrote", out)
    return table


if __name__ == "__main__":
    main()