"""Parameter sweeps of the tracker against the ground truth (step 6.9): one parameter at a time around the defaults.

Parameters: sigma_a (acceleration noise), sigma_extent (centre measurement noise along the line of sight, m), gate (association
chi-square), sigma_px (pixel noise of a camera bearing) and sigma_extra (its pose/calibration part, rad); the last two need the camera, slow. Each run is summarised in one row of the
metrics below, over all hazards of the recording.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from server.tracking.bearing import BearingConfig
from server.tracking.noise import MeasurementNoise
from testing.support.fusion_study import MOVING_MPS, STILL_MPS, replay_fusion, summarise_outputs

PARAMETERS = ("sigma_a", "sigma_extent", "gate", "sigma_px", "sigma_extra")
DEFAULTS = dict(sigma_a=1.4, sigma_extent=0.4, gate=18.4, sigma_px=2.0, sigma_extra=0.002)
METRICS = ["rows", "vel_rms_moving", "vel_rms_still", "phantom_rms", "static_while_moving_max", "static_while_still_mean",
           "leave_latency_max_s", "pos_rms", "nis"]


def _rms(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")


def _nanagg(f, x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(f(x)) if len(x) else float("nan")


def metrics(run: dict) -> dict:
    """One row of the sweep: velocity error of movers and of hazards that stand still (phantom speed), the worst
    static-while-moving rate of any mover (the dangerous error), the mean static-while-still rate, the slowest leave
    latency, the position error and the camera NIS."""
    rows = run["rows"]
    if rows.empty:
        return {m: float("nan") for m in METRICS} | dict(rows=0)
    mov, still = rows[rows["true_speed"] > MOVING_MPS], rows[rows["true_speed"] <= STILL_MPS]
    out = summarise_outputs(rows)
    return dict(rows=len(rows), vel_rms_moving=_rms(mov["vel_err"]), vel_rms_still=_rms(still["vel_err"]),
                phantom_rms=_rms(still["speed"]), static_while_moving_max=_nanagg(np.max, out["static_while_moving"]),
                static_while_still_mean=_nanagg(np.mean, out["static_while_still"]),
                leave_latency_max_s=_nanagg(np.max, out["leave_latency_s"]), pos_rms=_rms(rows["pos_err"]), nis=run["nis_mean"])


def run_one(folder, scans, params: dict, camera_every: int = 1, burn_in: int = 10, bearing: BearingConfig = BearingConfig()) -> dict:
    p = DEFAULTS | params
    use_camera = "sigma_px" in params or "sigma_extra" in params
    return replay_fusion(folder, scans, use_camera, gate_chi2=p["gate"], sigma_a=p["sigma_a"],
                         bearing=replace(bearing, sigma_px=p["sigma_px"], sigma_extra_rad=p["sigma_extra"]), camera_every=camera_every, burn_in=burn_in,
                         noise=replace(MeasurementNoise(), sigma_extent_m=p["sigma_extent"]))


def sweep(folder, scans, grid: dict, camera_every: int = 1, burn_in: int = 10, bearing: BearingConfig = BearingConfig()) -> pd.DataFrame:
    """grid: {parameter: [values]}. One row per value; the others stay at their defaults (the default is always included
    as the first row, parameter 'default')."""
    for k in grid:
        if k not in PARAMETERS:
            raise ValueError(f"unknown parameter {k!r}; choose from {PARAMETERS}")
    rows = [dict(parameter="default", value=float("nan")) | metrics(run_one(folder, scans, {}, camera_every, burn_in, bearing))]
    for k, values in grid.items():
        for v in values:
            rows.append(dict(parameter=k, value=float(v)) | metrics(run_one(folder, scans, {k: float(v)}, camera_every, burn_in, bearing)))
    return pd.DataFrame(rows, columns=["parameter", "value"] + METRICS)