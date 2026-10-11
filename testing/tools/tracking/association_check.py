"""Tracker check (steps 6.4, 6.5): gate, Hungarian assignment, lifecycle and merge against the ground truth.

    python -m testing.tools.tracking.association_check --recording dev_default_seed12345
    python -m testing.tools.tracking.association_check --recording dev_default_seed12345 --camera --gates 5.99,9.21,13.8
    python -m testing.tools.tracking.association_check --recording dev_default_seed12345 --point nearest

--recording   folder name inside testing/data/recordings, or a path (needs the ground-truth files)
--ground      range_image (default) or height;   --camera  use the camera edge refinement;   --scans A:B:S
--point       measurement point of the tracker: centre (default), nearest or centroid
--gates       squared Mahalanobis gates to compare (default 5.99,9.21,13.8,20)
--sigma-a     acceleration noise of the filter, m/s^2 (default 1.5)
--confirm     consecutive hits that confirm a track (default 3);  --max-misses  misses that delete one (default 5)
--out         output folder, default testing/data/association/<recording>

Writes summary.txt (printed too) with one row per gate and events.txt with the identity switches and duplicate tracks of
the gate 9.21. The columns are explained at the top of testing/support/association_study.py.
"""
import argparse
from pathlib import Path

import pandas as pd

from server.perception.ground import GroundConfig, GroundMethod
from server.perception.pipeline import PerceptionConfig
from server.tracking.track import TrackConfig
from testing.support.association_study import replay, scan_measurements
from testing.support.paths import DATA_DIR
from testing.support.replay_scans import resolve

COLUMNS = ["gate_chi2", "measurements", "correct", "switches", "fragments", "duplicates", "born", "confirmed", "merged",
           "coverage", "d2_mean", "d2_p95"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--ground", default="range_image", choices=[m.value for m in GroundMethod])
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--scans", default=None)
    ap.add_argument("--point", default="centre", choices=["centre", "nearest", "centroid"])
    ap.add_argument("--gates", default="5.99,9.21,13.8,20")
    ap.add_argument("--sigma-a", type=float, default=1.5)
    ap.add_argument("--confirm", type=int, default=3)
    ap.add_argument("--max-misses", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    folder = resolve(args.recording)
    scans = None
    if args.scans:
        a, b, s = (int(v) for v in args.scans.split(":"))
        scans = range(a, b, s)
    data = scan_measurements(folder, PerceptionConfig(ground=GroundConfig(method=GroundMethod(args.ground))), scans, args.camera)
    track = TrackConfig(confirm_hits=args.confirm, max_misses=args.max_misses)
    rows, events = [], []
    for g in (float(v) for v in args.gates.split(",")):
        s = replay(data, g, point=args.point, sigma_a=args.sigma_a, track=track)
        if abs(g - 9.21) < 1e-9:
            events = s["events"]
        rows.append({k: v for k, v in s.items() if k != "events"})
    table = pd.DataFrame(rows)[COLUMNS]
    text = (f"recording {folder.name}, point {args.point}, {len(data)} scans, confirm {args.confirm}, max misses {args.max_misses}\n\n"
            + table.round(2).to_string(index=False) + "\n")
    print(text)
    out = Path(args.out) if args.out else DATA_DIR / "association" / folder.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text(text)
    ev = pd.DataFrame(events, columns=["t", "kind", "hazard", "track_hazard", "d2"])
    (out / "events.txt").write_text("identity switches and duplicate tracks at gate 9.21\n"
                                    + (ev.round(2).to_string(index=False) if len(ev) else "none") + "\n")
    print("wrote", out)
    return table


if __name__ == "__main__":
    main()