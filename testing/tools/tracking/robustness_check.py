"""Robustness of the tracker against a slow processor, lost scans and a noisy or biased pose of the car (step 6.10).

    python -m testing.tools.tracking.robustness_check --recording dev_default_seed12345
    python -m testing.tools.tracking.robustness_check --recording dev_default_seed12345 --every 1,2,4,8,12 --repeats 5

--recording       folder name inside testing/data/recordings, or a path (needs ground-truth files);  --ground;  --scans A:B:S
--every           process only every n-th scan (default 1,2,3,5,8);  --drop  probability that a scan is lost (default 0.2,0.4)
--pos-sigma       pose position noise, m, per scan (default 0.05,0.1,0.2)
--yaw-sigma-deg   pose yaw noise, degrees, per scan (default 0.2,0.5,1)
--bias-pos        constant pose position error along x, m (default 0.2,0.5)
--bias-yaw-deg    constant pose yaw error, degrees (default 0.5,1,2)
--no-model-pose   do not also run the pose-noise cases with the noise told to the tracker (default: both, 'modelled' False/True)
--repeats         runs with different random seeds for the random cases, averaged (default 3);   --seed;   --burn-in
--out             output folder, default testing/data/robustness/<recording>

Each row: the scenario, the processing rate (hz) and the metrics of the tuning study (rows scored, velocity error of movers and of
hazards that stand still, static flags, position error) and identity switches, fragments and coverage of the association study.
Compare with the 'baseline' row. A yaw error moves a point at range r sideways by r * yaw, so it hurts far hazards most; a constant
bias only matters for the velocity when the car moves. LiDAR only. Writes summary.txt and summary.csv.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np

from server.perception.ground import GroundConfig, GroundMethod
from server.perception.pipeline import PerceptionConfig
from testing.support.association_study import scan_measurements
from testing.support.paths import DATA_DIR
from testing.support.replay_scans import resolve
from testing.support.robustness_study import run_scenarios, scenarios


def _floats(s: str) -> list:
    return [float(v) for v in s.split(",") if v.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--ground", default="range_image", choices=[m.value for m in GroundMethod])
    ap.add_argument("--scans", default=None)
    ap.add_argument("--every", default="1,2,3,5,8")
    ap.add_argument("--drop", default="0.2,0.4")
    ap.add_argument("--pos-sigma", default="0.05,0.1,0.2")
    ap.add_argument("--yaw-sigma-deg", default="0.2,0.5,1")
    ap.add_argument("--bias-pos", default="0.2,0.5")
    ap.add_argument("--bias-yaw-deg", default="0.5,1,2")
    ap.add_argument("--no-model-pose", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--burn-in", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    folder = resolve(args.recording)
    scans = None
    if args.scans:
        a, b, s = (int(v) for v in args.scans.split(":"))
        scans = range(a, b, s)
    data = scan_measurements(folder, PerceptionConfig(ground=GroundConfig(method=GroundMethod(args.ground))), scans, use_camera=True)
    scs = scenarios(every=[int(v) for v in _floats(args.every) if v > 1], drop=_floats(args.drop), pos_sigma=_floats(args.pos_sigma),
                    yaw_sigma=np.radians(_floats(args.yaw_sigma_deg)), bias_pos=_floats(args.bias_pos),
                    bias_yaw=np.radians(_floats(args.bias_yaw_deg)), model_pose=not args.no_model_pose)
    table = run_scenarios(folder, data, scs, args.repeats, args.seed, args.burn_in)
    text = (f"recording {folder.name}, {len(data)} scans, tuned parameters of 6.9, LiDAR only, {args.repeats} repeats for random cases\n\n"
            + table.round(3).to_string(index=False) + "\n")
    print(text)
    out = Path(args.out) if args.out else DATA_DIR / "robustness" / folder.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text(text)
    table.to_csv(out / "summary.csv", index=False)
    print("wrote", out)
    return table


if __name__ == "__main__":
    main()