"""Collision probability by sampling, with the deterministic fallback (step 7.7; section 5.6, "secondary signal").

The probability is a signal for the dashboard and the log. The caps of the stopping model do not depend on it.

Sampling. For every hazard (the tracks of `hazards.is_hazard`, PredictedTrack at the time of effect) N pairs of position and velocity are
drawn from the track's covariance (a track flagged static keeps zero velocity), and every sample gets a constant acceleration drawn from
N(0, accel_sigma^2) per axis (not for a static track); the reachable set of 7.3 is the worst case of this, the samples are the typical
case. Each sample moves its visible surface (`PredictedTrack.surface`, sampled along the edges) with constant acceleration over the
horizon. The car keeps its present speed and its path; its body is the rectangle [rear, front] x [-half width, half width] in the
coordinates of the path, widened by `base_radius_m` (the size of what is at a surface point). A sample hits when one of its surface points
is inside the rectangle at the same time step. p_i = hits / N; the total is 1 - prod(1 - p_i) (independent tracks).

Fallback. The deterministic check is the covariance-inflated conflict test of 7.3 (`find_conflicts`, sigma_scale = kappa z): p_i is 1 for a
track with a conflict and 0 without. It is used for everything when `force_deterministic` is set, and for the tracks that are left when the
time budget of the sampling is used up (the nearest hazards are sampled first). `method` says which: "sampled", "deterministic", "mixed".
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from server.risk.ego_path import Corridor, EgoGeometry, path_coordinates
from server.risk.hazards import HazardConfig, densify, find_conflicts, is_hazard
from server.risk.timing import PredictedTrack


@dataclass(frozen=True)
class CollisionConfig:
    samples: int = 200
    horizon_s: float = 4.0
    step_s: float = 0.1
    accel_sigma_mps2: float = 0.5       # per axis, typical acceleration of an object (a_obj of 7.3 is the worst case)
    surface_spacing_m: float = 0.5
    base_radius_m: float = 0.2
    time_budget_s: float = 0.008        # of the 33 ms cycle; what is left after it is answered by the deterministic check
    max_tracks: int = 12
    seed: int = 0
    force_deterministic: bool = False


@dataclass(frozen=True)
class CollisionResult:
    p_total: float
    per_track: dict                     # track id -> probability (0..1)
    method: str                         # "sampled", "deterministic" or "mixed"
    sampled_ids: tuple = ()
    elapsed_s: float = 0.0


def _surface(track: PredictedTrack, spacing: float) -> np.ndarray:
    if track.surface is None:
        return np.zeros((1, 2))
    s = np.asarray(track.surface, dtype=float)
    return np.vstack([densify(s[:4], spacing), s[4:5]])


def _draw(mean: np.ndarray, cov: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    c = 0.5 * (cov + cov.T) + 1e-9 * np.eye(len(mean))
    try:
        L = np.linalg.cholesky(c)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(c)
        L = v * np.sqrt(np.maximum(w, 0.0))
    return mean + rng.standard_normal((n, len(mean))) @ L.T


def hit_probability(track: PredictedTrack, corridor: Corridor, ego_speed_mps: float, ego_travel_m: float, geometry: EgoGeometry,
                    cfg: CollisionConfig, rng: np.random.Generator) -> float:
    """Fraction of the sampled futures of one track in which it meets the body of the car."""
    n = cfg.samples
    taus = np.arange(0.0, cfg.horizon_s + 1e-9, cfg.step_s)
    state = _draw(np.concatenate([np.asarray(track.position, float), np.asarray(track.velocity, float)]), track.cov, n, rng)
    pos, vel = state[:, :2], state[:, 2:]
    acc = np.zeros((n, 2))
    if track.static:
        vel = np.zeros((n, 2))
    else:
        acc = rng.standard_normal((n, 2)) * cfg.accel_sigma_mps2
    offs = _surface(track, cfg.surface_spacing_m)                                        # (P, 2)
    t = taus[None, :, None]
    centre = pos[:, None, :] + vel[:, None, :] * t + 0.5 * acc[:, None, :] * t * t       # (N, T, 2)
    pts = centre[:, :, None, :] + offs[None, None, :, :]                                 # (N, T, P, 2)
    s, lat = path_coordinates(pts.reshape(-1, 2), corridor.k)
    s, lat = s.reshape(pts.shape[:3]), lat.reshape(pts.shape[:3])
    front = ego_travel_m + ego_speed_mps * taus + geometry.front_offset_m                # (T,)
    rear = ego_travel_m + ego_speed_mps * taus - geometry.rear_overhang_m
    r = cfg.base_radius_m
    inside = (s >= rear[None, :, None] - r) & (s <= front[None, :, None] + r) & (np.abs(lat) <= corridor.half_width_m + r)
    return float(inside.any(axis=(1, 2)).mean())


def collision_probability(tracks: list, corridor: Corridor, ego_speed_mps: float, ego_travel_m: float = 0.0,
                          geometry: EgoGeometry = EgoGeometry(), cfg: CollisionConfig = CollisionConfig(),
                          hazard_cfg: HazardConfig = HazardConfig(), sigma_scale: float = 1.0, rng: np.random.Generator | None = None,
                          clock=time.perf_counter) -> CollisionResult:
    """Probability that the car, keeping its speed and path, meets one of the hazards among `tracks` within the horizon."""
    t0 = clock()
    rng = rng if rng is not None else np.random.default_rng(cfg.seed)
    hazards = [tr for tr in tracks if is_hazard(tr, hazard_cfg)]
    hazards.sort(key=lambda tr: float(np.hypot(*(np.asarray(tr.position, float)))))                      # nearest first (vehicle frame)
    per, sampled, rest = {}, [], []
    for i, tr in enumerate(hazards):
        if cfg.force_deterministic or i >= cfg.max_tracks or (sampled and clock() - t0 > cfg.time_budget_s):
            rest.append(tr)
            continue
        per[tr.id] = hit_probability(tr, corridor, ego_speed_mps, ego_travel_m, geometry, cfg, rng)
        sampled.append(tr.id)
    if rest:
        det = {c.track_id for c in find_conflicts(rest, corridor, ego_speed_mps, ego_travel_m, geometry, hazard_cfg, sigma_scale)}
        for tr in rest:
            per[tr.id] = 1.0 if tr.id in det else 0.0
    method = "deterministic" if not sampled and rest else ("mixed" if rest else "sampled")
    if not hazards:
        method = "sampled"
    p_total = 1.0 - math.prod(1.0 - p for p in per.values())
    return CollisionResult(p_total=p_total, per_track=per, method=method, sampled_ids=tuple(sampled), elapsed_s=clock() - t0)