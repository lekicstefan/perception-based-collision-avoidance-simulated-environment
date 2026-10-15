"""Hazard prediction, the reachable set and the time-of-arrival conflict test (step 7.3; section 5.6).

Nothing here knows what an object is. Every track obeys one physical rule: over the horizon its velocity can change by at most a_obj
(default 2 m/s^2). The surface that can be hit is the visible box of its last cluster and its nearest point (the tracked centre is biased
along the line of sight and is not used for distances), moved with the track. Its possible position after tau seconds (counted from the
time of effect) is therefore the nominal one (constant velocity, zero for a track flagged static) plus a radius

    r(tau) = r_base + a_obj tau^2 / 2 + kappa z_eps sqrt(lambda_max(C(tau)))             C(tau) = C_pp + tau (C_pv + C_vp) + tau^2 C_vv

around every point of that surface (the process noise is not in C: a_obj stands for it; kappa z_eps is the inflation and quantile of the
risk level, `sigma_scale`). The reachable set is evaluated every `step_s` up to `horizon_s`.

Conflict test (time of arrival). In the coordinates of the path (arc length s, lateral offset l; ego_path.py) a disc of radius r around a
point at (s_i, l_i) meets the corridor |l| <= half width when r >= delta_i = max(|l_i| - half width, 0), over the arc lengths
s_i -+ sqrt(r^2 - delta_i^2). Per slice this gives an interval [a, b] of the path (the hull over the points).

  * conflict point  the nearest arc length a of the first slice (the smallest tau) in which the reachable set meets the corridor;
                    the distance the car can still drive is d_path = a - front offset - the distance the car drives until the time of effect;
  * segment         [a, b] of that slice, tau_first = its tau (the earliest time the reachable set can be there);
  * clearing time   the time t_clear in which the rear of the car has passed b at its present speed;
  * no conflict     when t_clear + `time_margin_s` < tau_first (the car is past before anything can be there) or when the segment is already
                    behind the car; otherwise the hazard is treated as stationary at the conflict point (the stopping model of stopping.py).
A car that stands still never clears: every hazard that can reach the corridor within the horizon is a conflict.
The same test covers a pedestrian who could still cross (the car arrives first or not) and one walking away, without any class.

A conflict also carries what the stopping model needs: the speed of the hazard along the path and its standard deviation (lead vehicle
allowance), sigma_d of the distance, and a margin weight: confirmed and coasting tracks count fully, a tentative track (few hits, may be a
false alarm) counts with `tentative_margin_weight`, so that it is a hazard from its first scans but earns a smaller safety margin until
it is confirmed (the caps use the surface distance and the stopping model either way).
Known simplifications: the region of a track is a union of discs on its visible surface (the unseen back of an object is not part of
it); the covariance term is in the region and also in d_eff (the distance uncertainty), a small double count that the sensitivity
analysis (7.9) shows; the speed of the car is held constant over the horizon.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from server.risk.ego_path import Corridor, EgoGeometry, path_coordinates
from server.risk.timing import PredictedTrack
from server.risk.uncertainty import distance_sigma, path_tangent, speed_along_path


@dataclass(frozen=True)
class HazardConfig:
    horizon_s: float = 4.0
    step_s: float = 0.1
    max_object_accel: float = 2.0       # a_obj (m/s^2)
    base_radius_m: float = 0.2          # floor for the size of whatever is at a surface point
    edge_spacing_m: float = 0.25        # the edges of the visible box are sampled at this spacing
    time_margin_s: float = 0.5          # the car must be past the segment this much before the hazard can be there
    include_tentative: bool = True
    tentative_margin_weight: float = 0.5
    min_clear_speed_mps: float = 0.1    # a slower car does not clear a segment
    nominal_velocity_sigma_k: float = 2.0   # the NOMINAL motion (in_corridor) uses the estimated speed reduced by this many standard deviations of the velocity:
                                            # a standing object whose velocity estimate is noise (0.4 m/s +- 0.2) is not "walking into the lane"


@dataclass(frozen=True)
class Conflict:
    track_id: int
    state: str
    static: bool
    tau_first_s: float
    s_conflict_m: float             # arc length of the conflict point (path of the data-time frame)
    d_path_m: float                 # the distance the front of the car can still drive to the conflict point at the time of effect
    segment_m: tuple                # (a, b) of the first slice
    t_clear_s: float                # inf for a car that stands still
    v_along_mps: float              # speed of the hazard along the path direction at the conflict point (nominal, 0 for a static track)
    v_sigma_mps: float
    sigma_d_m: float
    margin_weight: float
    existence: float
    in_corridor: bool = True        # the NOMINAL motion (no reachable-set growth, no covariance) of the object enters the corridor; False: the conflict
                                    # exists only because the object could (worst case) get there: a precaution, answered with a floor speed (decision.py)


def is_hazard(track: PredictedTrack, cfg: HazardConfig = HazardConfig()) -> bool:
    return track.state in ("confirmed", "coasting") or (track.state == "tentative" and cfg.include_tentative)


def margin_weight(track: PredictedTrack, cfg: HazardConfig = HazardConfig()) -> float:
    """Weight of the safety margin of a hazard: 1 for a confirmed or coasting track, `tentative_margin_weight` for a tentative one."""
    return cfg.tentative_margin_weight if track.state == "tentative" else 1.0


def densify(points: np.ndarray, spacing: float) -> np.ndarray:
    """Points along the closed polygon through `points` (n, 2) with at most `spacing` between neighbours (corners included)."""
    pts = np.asarray(points, dtype=float)
    out = []
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        n = max(int(math.ceil(float(np.hypot(*(b - a))) / spacing)), 1)
        out.append(a + (b - a) * (np.arange(n)[:, None] / n))
    return np.vstack(out)


def footprint_offsets(track: PredictedTrack, cfg: HazardConfig = HazardConfig()) -> np.ndarray:
    """Offsets (m, 2) from the tracked point to the sampled surface points: the edges of the visible box and the nearest point."""
    if track.surface is None:
        return np.zeros((1, 2))
    s = np.asarray(track.surface, dtype=float)
    return np.vstack([densify(s[:4], cfg.edge_spacing_m), s[4:5]])


def _sigma_pos(track: PredictedTrack, tau: float) -> float:
    """Largest standard deviation of the position at tau seconds from the time of effect (no process noise)."""
    c = track.cov
    m = c[:2, :2] + tau * (c[:2, 2:] + c[2:, :2]) + tau * tau * c[2:, 2:]
    return math.sqrt(max(float(np.linalg.eigvalsh(0.5 * (m + m.T))[-1]), 0.0))


def reachable_radius(track: PredictedTrack, tau: float, sigma_scale: float, cfg: HazardConfig = HazardConfig()) -> float:
    return cfg.base_radius_m + 0.5 * cfg.max_object_accel * tau * tau + sigma_scale * _sigma_pos(track, tau)


def corridor_interval(centres: np.ndarray, radius: float, corridor: Corridor):
    """(a, b) arc lengths that the discs (radius around `centres` (n, 2)) cover inside the corridor, or None."""
    s, lat = path_coordinates(centres, corridor.k)
    delta = np.maximum(np.abs(lat) - corridor.half_width_m, 0.0)
    hit = delta <= radius
    if not hit.any():
        return None
    half = np.sqrt(np.maximum(radius * radius - delta[hit] ** 2, 0.0))
    return float(np.min(s[hit] - half)), float(np.max(s[hit] + half))


def nominal_velocity(track: PredictedTrack, vel: np.ndarray, k: float) -> np.ndarray:
    """The estimated velocity with its speed reduced by k standard deviations (not below zero): the part of the motion that is significant."""
    speed = float(np.hypot(*vel))
    if speed < 1e-9 or k <= 0.0:
        return vel
    sigma = math.sqrt(max(float(np.linalg.eigvalsh(0.5 * (track.vel_cov + track.vel_cov.T))[-1]), 0.0))
    return vel * (max(speed - k * sigma, 0.0) / speed)


def nominal_in_corridor(centres: np.ndarray, vel: np.ndarray, taus: np.ndarray, corridor: Corridor, tail0: float,
                        cfg: HazardConfig = HazardConfig()) -> bool:
    """Does the object, moving at its estimated velocity (zero for a static one) and with only the base radius around its surface points, enter the
    corridor ahead of the rear of the car within the horizon?"""
    for tau in taus:
        iv = corridor_interval(centres + vel * tau, cfg.base_radius_m, corridor)
        if iv is not None and iv[1] >= tail0 and iv[0] <= corridor.length_m:
            return True
    return False


def find_conflicts(tracks: list, corridor: Corridor, ego_speed_mps: float, ego_travel_m: float = 0.0,
                   geometry: EgoGeometry = EgoGeometry(), cfg: HazardConfig = HazardConfig(), sigma_scale: float = 1.0,
                   ego_sigma_m: float = 0.0) -> list:
    """The conflicts of the hazards among `tracks` (PredictedTrack at the time of effect) with the corridor of the car, nearest first.
    `ego_travel_m` is the distance the car drives between the data time and the time of effect (the tracks are already at the latter),
    `sigma_scale` is kappa * z_eps of the risk level, `ego_sigma_m` the position uncertainty of the car's own pose."""
    taus = np.arange(0.0, cfg.horizon_s + 1e-9, cfg.step_s)
    tail0 = ego_travel_m - geometry.rear_overhang_m                          # arc length of the rear of the car at the time of effect
    out = []
    for tr in tracks:
        if not is_hazard(tr, cfg):
            continue
        offs = footprint_offsets(tr, cfg)
        vel = np.zeros(2) if tr.static else np.asarray(tr.velocity, dtype=float)
        first = None
        for tau in taus:
            iv = corridor_interval(tr.position + offs + vel * tau, reachable_radius(tr, tau, sigma_scale, cfg), corridor)
            if iv is None or iv[1] < tail0 or iv[0] > corridor.length_m:         # not in the corridor, behind the car, or beyond what is looked at
                continue
            first = (float(tau), iv)
            break
        if first is None:
            continue
        tau_first, (a, b) = first
        t_clear = math.inf if ego_speed_mps < cfg.min_clear_speed_mps else max(b - tail0, 0.0) / ego_speed_mps
        if t_clear + cfg.time_margin_s < tau_first:
            continue                                                             # the car is past the segment before anything can be there
        tangent = path_tangent(a, corridor.k)
        v_h, v_sig = speed_along_path(vel, tr.vel_cov, tangent)
        out.append(Conflict(
            track_id=tr.id, state=tr.state, static=tr.static, tau_first_s=tau_first, s_conflict_m=a,
            d_path_m=max(a - corridor.front_offset_m - ego_travel_m, 0.0), segment_m=(a, b), t_clear_s=t_clear, v_along_mps=v_h,
            v_sigma_mps=v_sig, sigma_d_m=distance_sigma(tr.pos_cov, tangent, ego_sigma_m), margin_weight=margin_weight(tr, cfg),
            existence=tr.existence, in_corridor=nominal_in_corridor(tr.position + offs, nominal_velocity(tr, vel, cfg.nominal_velocity_sigma_k), taus, corridor, tail0, cfg)))
    return sorted(out, key=lambda c: c.d_path_m)