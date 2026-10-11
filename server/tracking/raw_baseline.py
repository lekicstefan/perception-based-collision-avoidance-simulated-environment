"""Baseline velocity estimator: raw two-frame differencing (step 6.8), kept for the filter-versus-raw ablation.

    v = (z_k - z_{k-1}) / (t_k - t_{k-1})

z is the measured position (the same point of the cluster the filter uses) of two consecutive measurements assigned to the
same track. There is no model and no smoothing: every error of the measured point becomes velocity error, divided by the
time between the scans (at 10 Hz a position error of 0.1 m is a velocity error of 1 m/s). Everything else (association,
lifecycle) is the same as in the tracker, so the comparison isolates the estimator.

`raw_static` is the naive static test that goes with it: the speed below a fixed threshold, no uncertainty used.
"""
from __future__ import annotations

import numpy as np


def raw_velocity(z_prev, t_prev: float, z, t: float):
    """Two-frame velocity (2,) or None when the time stamps do not advance."""
    dt = float(t) - float(t_prev)
    if dt <= 1e-6:
        return None
    return (np.asarray(z, dtype=float) - np.asarray(z_prev, dtype=float)) / dt


def raw_static(v, threshold_mps: float = 0.5) -> bool:
    return bool(np.hypot(*v) < threshold_mps)