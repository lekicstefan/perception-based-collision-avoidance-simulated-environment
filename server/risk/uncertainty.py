"""Uncertainty of the distance and of the speed along the path (step 7.6).

The effective distance of the stopping model, d_eff = d_path - margin - z_eps kappa sigma_d + ..., needs sigma_d: the standard deviation
of the distance from the car to the conflict point. A distance along the path moves with a position error along the path direction u at
the conflict point (the tangent of the arc), so with the position covariance C of the track (vehicle frame):

    sigma_d^2 = u^T C u + sigma_ego^2

where sigma_ego is the position uncertainty of the car's own pose (the pose noise of the simulation, which the tracker has to be told too,
phase 6 finding 1). The quantile z_eps of the acceptable risk, the covariance inflation kappa and the margin belong to the risk level (7.8)
and are applied in `stopping.effective_distance`; the covariance describes measurement and process noise only, so kappa covers the
systematic errors (latency, segmentation bias, sudden manoeuvres).

The hazard braking allowance of a lead needs the speed of the hazard along the path, v_h = v . u, and its standard deviation
sigma_v = sqrt(u^T C_v u), from which the lower quantile v_h,low = max(0, v_h - z_eps sigma_v) is taken (`stopping.lead_speed_low`).
"""
from __future__ import annotations

import math

import numpy as np


def path_tangent(s: float, k: float) -> np.ndarray:
    """Unit direction of the predicted path at arc length s (vehicle frame): the heading of the arc is k s."""
    return np.array([math.cos(k * s), math.sin(k * s)])


def distance_sigma(pos_cov, tangent, ego_sigma_m: float = 0.0) -> float:
    """sigma_d (m): standard deviation of the distance along the path to a point whose position has covariance pos_cov (2, 2)."""
    u = np.asarray(tangent, dtype=float)
    c = np.asarray(pos_cov, dtype=float)
    return math.sqrt(max(float(u @ c @ u), 0.0) + float(ego_sigma_m) ** 2)


def speed_along_path(velocity, vel_cov, tangent) -> tuple:
    """(v_h, sigma_v): speed of the hazard along the path direction (positive = moving the way the car drives) and its standard deviation."""
    u = np.asarray(tangent, dtype=float)
    return float(np.asarray(velocity, dtype=float) @ u), math.sqrt(max(float(u @ np.asarray(vel_cov, dtype=float) @ u), 0.0))