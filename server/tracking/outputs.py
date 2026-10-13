"""Derived outputs of a track (step 6.7): what the risk stage gets to see.

Everything here is computed from the state and covariance of the filter and from the history of the track; nothing depends on
what kind of object it is.

  velocity, speed, heading   from the filter; the heading (direction of the velocity) exists only for a moving track
  speed_sigma, heading_sigma the uncertainty of the speed and the heading, propagated from the velocity covariance
  position covariance        and its 95 % ellipse (semi-major, semi-minor, angle of the major axis)
  age, hits, misses, hit ratio, camera hits
  existence                  share of the last `existence_window` LiDAR scans in which the track got a measurement (a track of
                             few scans is divided by the scans it has lived)
  static / dynamic flag      static means "known not to move":
        1. the velocity is known: its largest standard deviation is below `known_sigma_mps` (a new track is not static: its
           velocity is unknown, so it is treated as able to move),
        2. the velocity is not significantly different from zero (the squared Mahalanobis distance of v from 0 is below
           `static_chi2`, chi-square 2 dof, 95 % = 5.99) and its speed is below `static_max_speed_mps`,
        3. this has held for `static_confirm_scans` scans in a row.
     It is left as soon as the zero-velocity test fails at the stricter level `leave_chi2` (99.9 %) or the speed exceeds the
     limit. The asymmetry keeps the flag from flickering and makes it quick to see motion and slow to believe stillness.
  velocity_out               the velocity with the phantom velocity of a static track removed (zero); dynamic tracks unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OutputConfig:
    known_sigma_mps: float = 0.9            # velocity standard deviation (largest axis) below which the velocity counts as known
    static_chi2: float = 5.99               # zero-velocity test to become static (95 %, 2 dof)
    leave_chi2: float = 10.5                # zero-velocity test failed at this level: static is left (99.9 %)
    static_max_speed_mps: float = 0.6       # a faster track is never static
    static_confirm_scans: int = 4
    ellipse_chi2: float = 5.99              # confidence level of the position ellipse (95 %)


@dataclass(frozen=True)
class TrackOutput:
    id: int
    state: str
    t: float
    position: np.ndarray                    # (2,) world
    velocity: np.ndarray                    # (2,) estimate of the filter
    velocity_out: np.ndarray                # (2,) zero for a static track
    speed: float
    speed_sigma: float
    heading: float | None                   # rad from the world x axis, positive to the left; None for a static track
    heading_sigma: float | None
    pos_cov: np.ndarray                     # (2, 2)
    ellipse: tuple                          # (semi-major m, semi-minor m, angle of the major axis rad) at ellipse_chi2
    cov: np.ndarray                         # (4, 4) full state covariance
    velocity_known: bool
    static: bool
    age_scans: int
    age_s: float
    hits: int
    misses: int
    hit_ratio: float
    camera_hits: int
    existence: float
    surface_offsets: np.ndarray | None = None   # (5, 2): corners of the visible box and nearest point, relative to `position` (world axes)


def velocity_stats(v, p_vv):
    """(speed, speed standard deviation, heading standard deviation or None) from the velocity and its 2x2 covariance."""
    v = np.asarray(v, dtype=float)
    speed = float(np.hypot(*v))
    if speed > 1e-3:
        u = v / speed
        perp = np.array([-u[1], u[0]])
        return speed, float(math.sqrt(max(u @ p_vv @ u, 0.0))), float(math.sqrt(max(perp @ p_vv @ perp, 0.0)) / speed)
    return speed, float(math.sqrt(max(np.linalg.eigvalsh(p_vv)[-1], 0.0))), None


def zero_velocity_d2(v, p_vv) -> float:
    """Squared Mahalanobis distance of the velocity from zero."""
    v = np.asarray(v, dtype=float)
    return float(v @ np.linalg.solve(p_vv, v))


def update_motion_state(track, cfg: OutputConfig = OutputConfig()) -> None:
    """Advance the static flag of a track by one scan (see the module docstring)."""
    f = track.filter
    p_vv = f.P[2:, 2:]
    speed = f.speed
    known = float(np.linalg.eigvalsh(p_vv)[-1]) <= cfg.known_sigma_mps ** 2
    d2 = zero_velocity_d2(f.velocity, p_vv)
    if track.static:
        if d2 > cfg.leave_chi2 or speed > cfg.static_max_speed_mps:
            track.static, track.static_count = False, 0
        return
    if known and d2 <= cfg.static_chi2 and speed <= cfg.static_max_speed_mps:
        track.static_count += 1
        if track.static_count >= cfg.static_confirm_scans:
            track.static = True
    else:
        track.static_count = 0


def track_output(track, cfg: OutputConfig = OutputConfig()) -> TrackOutput:
    f = track.filter
    p_vv = f.P[2:, 2:]
    speed, speed_sigma, heading_sigma = velocity_stats(f.velocity, p_vv)
    w, vec = np.linalg.eigh(f.P[:2, :2])
    major, minor = (math.sqrt(max(cfg.ellipse_chi2 * x, 0.0)) for x in (w[1], w[0]))
    known = float(np.linalg.eigvalsh(p_vv)[-1]) <= cfg.known_sigma_mps ** 2
    return TrackOutput(
        id=track.id, state=track.state.value, t=f.t, position=f.position, velocity=f.velocity,
        velocity_out=np.zeros(2) if track.static else f.velocity, speed=speed, speed_sigma=speed_sigma,
        heading=None if track.static else f.heading, heading_sigma=None if track.static else heading_sigma,
        pos_cov=f.P[:2, :2].copy(), ellipse=(major, minor, float(math.atan2(vec[1, 1], vec[0, 1]))), cov=f.P.copy(),
        velocity_known=known, static=track.static, age_scans=track.age_scans, age_s=f.t - track.born_t, hits=track.hits,
        misses=track.misses, hit_ratio=track.hit_ratio, camera_hits=track.camera_hits, existence=track.hit_ratio,
        surface_offsets=getattr(track, "surface_offsets", None))