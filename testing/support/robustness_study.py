"""Robustness of the tracker (step 6.10): how do the results change when the processor is slow or loses scans, and when the
pose of the car is noisy or biased?

The perception is run once (`scan_measurements`); the scans are then changed before they reach the tracker:

  rate         `thin_scans`: only every n-th scan is processed (a slower processor: the tracker works with the real time step
               of the scans) and/or every scan is lost with probability `drop` (a random gap in the stream)
  pose noise   `perturb_scans`: the pose of the car of every scan is wrong by a random position error (std `pos_sigma`, m,
               both axes) and yaw error (std `yaw_sigma`, rad), independent between scans
  pose bias    the same constant error in every scan: `bias_pos` (m, along the world x axis) and `bias_yaw` (rad)

A yaw error eps turns every measured point about the car by eps (a point at range r moves r*eps sideways), a position error
moves it as it is. The change is applied to every position of the world cluster, to its bearing and to its pose stamp.
The tracker can be told about the pose noise (`modelled`: MeasurementNoise.pose_position_sigma_m / pose_yaw_sigma_rad) or not.
Scores: the metrics of tuning_study (velocity error, static flags, rows scored) and, from the association study, identity
switches, fragments and coverage. Without the camera (its pose would need the same perturbation).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from server.tracking.association import AssociationConfig
from server.tracking.measurement import WorldCluster
from server.tracking.noise import MeasurementNoise
from testing.support.association_study import replay
from testing.support.fusion_study import replay_fusion
from testing.support.tuning_study import metrics

TUNED = dict(gate_chi2=18.4, sigma_a=1.4, sigma_extent_m=0.4)
COLUMNS = ["group", "name", "modelled", "hz", "rows", "vel_rms_moving", "vel_rms_still", "phantom_rms", "static_while_moving_max",
           "static_while_still_mean", "leave_latency_max_s", "pos_rms", "switches", "fragments", "coverage"]


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def shift_cluster(w: WorldCluster, e, eps: float) -> WorldCluster:
    """The cluster as it would be placed with a pose that is wrong by the position error e (2,) and the yaw error eps."""
    c = np.asarray(w.pose[:2], dtype=float)
    e = np.asarray(e, dtype=float)

    def f(p):
        v = np.asarray(p, dtype=float) - c
        return np.asarray(p, dtype=float) + e + eps * np.stack([-v[..., 1], v[..., 0]], axis=-1)

    pose = (w.pose[0] + e[0], w.pose[1] + e[1], w.pose[2], w.pose[3] + eps) + tuple(w.pose[4:])
    return replace(w, pose=pose, nearest_xy=f(w.nearest_xy), centre_xy=f(w.centre_xy), centroid_xy=f(w.centroid_xy),
                   sensor_xy=f(w.sensor_xy), corners_xy=f(w.corners_xy), bearing=_wrap(w.bearing + eps))


def perturb_scans(scans: list, pos_sigma: float = 0.0, yaw_sigma: float = 0.0, bias_pos: float = 0.0, bias_yaw: float = 0.0,
                  seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for t, items in scans:
        e = rng.normal(0.0, pos_sigma, 2) + np.array([bias_pos, 0.0])
        eps = float(rng.normal(0.0, yaw_sigma)) + bias_yaw
        out.append((t, [(shift_cluster(w, e, eps), h) for w, h in items]))
    return out


def thin_scans(scans: list, every: int = 1, drop: float = 0.0, seed: int = 0) -> list:
    """Every `every`-th scan, each lost with probability `drop` (the first scan is always kept)."""
    rng = np.random.default_rng(seed)
    kept = scans[::max(int(every), 1)]
    return [s for i, s in enumerate(kept) if i == 0 or rng.random() >= drop]


def rate_hz(scans: list) -> float:
    t = np.array([s[0] for s in scans])
    return float(1.0 / np.median(np.diff(t))) if len(t) > 2 else float("nan")


def evaluate(folder, scans: list, modelled: tuple = (0.0, 0.0), burn_in: int = 10) -> dict:
    """Metrics of one set of (changed) scans. `modelled` = (position sigma, yaw sigma) the tracker is told about."""
    noise = MeasurementNoise(sigma_extent_m=TUNED["sigma_extent_m"], pose_position_sigma_m=modelled[0], pose_yaw_sigma_rad=modelled[1])
    run = replay_fusion(folder, scans, False, gate_chi2=TUNED["gate_chi2"], sigma_a=TUNED["sigma_a"], burn_in=burn_in, noise=noise)
    m = metrics(run)
    st = replay(scans, TUNED["gate_chi2"], noise=noise, sigma_a=TUNED["sigma_a"])
    return dict(m, hz=rate_hz(scans), switches=st["switches"], fragments=st["fragments"], coverage=st["coverage"])


def scenarios(every=(), drop=(), pos_sigma=(), yaw_sigma=(), bias_pos=(), bias_yaw=(), model_pose: bool = True) -> list:
    """[(group, name, thin kwargs, perturb kwargs, modelled)] with a baseline first."""
    out = [("baseline", "baseline", {}, {}, False)]
    out += [("rate", f"every {n}", dict(every=n), {}, False) for n in every]
    out += [("rate", f"drop {p:g}", dict(drop=p), {}, False) for p in drop]
    for s in pos_sigma:
        out += [("pose noise", f"pos {s:g} m", {}, dict(pos_sigma=s), m) for m in ((False, True) if model_pose else (False,))]
    for s in yaw_sigma:
        out += [("pose noise", f"yaw {np.degrees(s):g} deg", {}, dict(yaw_sigma=s), m) for m in ((False, True) if model_pose else (False,))]
    out += [("pose bias", f"pos {b:g} m", {}, dict(bias_pos=b), False) for b in bias_pos]
    out += [("pose bias", f"yaw {np.degrees(b):g} deg", {}, dict(bias_yaw=b), False) for b in bias_yaw]
    return out


def run_scenarios(folder, scans: list, scs: list, repeats: int = 3, seed: int = 0, burn_in: int = 10) -> pd.DataFrame:
    """Every scenario `repeats` times with different seeds (only the random ones differ); the numbers are averaged."""
    rows = []
    for group, name, thin, pert, modelled in scs:
        random_run = bool(pert.get("pos_sigma") or pert.get("yaw_sigma") or thin.get("drop"))
        res = []
        for r in range(repeats if random_run else 1):
            s = thin_scans(scans, seed=seed + r, **thin)
            s = perturb_scans(s, seed=seed + 1000 + r, **pert)
            told = (pert.get("pos_sigma", 0.0), pert.get("yaw_sigma", 0.0)) if modelled else (0.0, 0.0)
            res.append(evaluate(folder, s, told, burn_in))
        row = {k: float(np.nanmean([x[k] for x in res])) for k in res[0]}
        rows.append(dict(group=group, name=name, modelled=bool(modelled), **row))
    return pd.DataFrame(rows, columns=COLUMNS)