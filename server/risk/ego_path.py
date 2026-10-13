"""Ego path and corridor (step 7.1): where the car will drive, as a constant-curvature arc.

The path is the arc the middle of the rear axle follows if the car keeps its curvature (the origin of the vehicle frame is the rear axle,
x forward, y left). It starts at the origin and has the signed curvature k (1/m, positive = left turn):

    steering      k = tan(delta) / wheelbase                 (kinematic bicycle; default: the steering angle is what the car is told to do)
    yaw rate      k = omega / v                              (what the car does; below `min_speed_mps` the steering is used instead)

The corridor is the strip of `width + 2 * lateral_margin` around that arc. Points are described in the coordinates of the path:

    s   arc length along the path (m) of the point on the arc nearest to it, from the rear axle (negative behind it)
    l   signed lateral offset (m) from the arc, positive to the left

For a conflict point at s the distance the car can still drive before its front reaches it is d = s - front_offset (the front bumper is
`front_offset` ahead of the rear axle; the corridor of a turning car is approximated by the strip around the axle arc, the lateral
margin covers the difference at the radii that occur, which are much larger than the car).
Not done here: the planned-route option of the specification (a real planner's intended path, an allowed alternative to the arc).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EgoGeometry:
    wheelbase_m: float = 2.7
    length_m: float = 4.4
    width_m: float = 1.8
    rear_overhang_m: float = 0.9

    @property
    def front_offset_m(self) -> float:
        """Distance of the front bumper ahead of the rear axle (the origin of the vehicle frame)."""
        return self.length_m - self.rear_overhang_m

    @classmethod
    def from_calibration(cls, ego) -> "EgoGeometry":
        """From the calibration's ego block (`Calibration.ego`: wheelbase, length, width, rear overhang)."""
        return cls(float(ego.wheelbase), float(ego.length), float(ego.width), float(ego.rear_overhang))


@dataclass(frozen=True)
class EgoState:
    """The car's own state as it arrives in the header of every sensor message."""
    speed_mps: float
    yaw_rate_rps: float = 0.0
    steering_rad: float = 0.0


@dataclass(frozen=True)
class EgoPathConfig:
    source: str = "steering"            # "steering" or "yaw_rate"
    min_speed_mps: float = 0.5          # below this the yaw rate says nothing about the curvature (k = omega / v) and the steering is used
    lateral_margin_m: float = 0.3       # the corridor is the car's width plus this on each side
    length_m: float = 100.0             # arc length of the path that is looked at (about the range of the LiDAR)
    max_curvature: float = 0.5          # |k| limit (radius 2 m): protects the arc geometry from a noisy yaw rate at low speed


def curvature(state: EgoState, geometry: EgoGeometry = EgoGeometry(), cfg: EgoPathConfig = EgoPathConfig()) -> float:
    """Signed curvature (1/m) of the predicted path."""
    if cfg.source == "yaw_rate" and abs(state.speed_mps) >= cfg.min_speed_mps:
        k = state.yaw_rate_rps / state.speed_mps
    elif cfg.source in ("steering", "yaw_rate"):
        k = math.tan(state.steering_rad) / geometry.wheelbase_m
    else:
        raise ValueError(f"unknown curvature source {cfg.source!r}")
    return float(np.clip(k, -cfg.max_curvature, cfg.max_curvature))


def path_coordinates(points_xy, k: float) -> tuple:
    """(s, l) of points (n, 2) or (2,) in the vehicle frame with respect to the arc of curvature k (see the module text)."""
    p = np.atleast_2d(np.asarray(points_xy, dtype=float))
    x, y = p[:, 0], p[:, 1]
    if abs(k) < 1e-9:
        s, lat = x.copy(), y.copy()
    else:
        m = 1.0 if k > 0 else -1.0                                    # a right turn is the mirror image of a left turn
        r = 1.0 / abs(k)
        ym = m * y
        s = np.arctan2(x, r - ym) * r
        lat = m * (r - np.hypot(x, r - ym))
    if np.ndim(points_xy) == 1:
        return float(s[0]), float(lat[0])
    return s, lat


def point_on_path(s, k: float) -> np.ndarray:
    """Positions (n, 2) of the arc at the arc lengths s."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    if abs(k) < 1e-9:
        return np.stack([s, np.zeros_like(s)], axis=1)
    return np.stack([np.sin(k * s) / k, (1.0 - np.cos(k * s)) / k], axis=1)


@dataclass(frozen=True)
class Corridor:
    """The strip around the predicted path."""
    k: float                    # signed curvature (1/m)
    half_width_m: float         # half of the car's width plus the lateral margin
    length_m: float
    front_offset_m: float

    def coordinates(self, points_xy) -> tuple:
        return path_coordinates(points_xy, self.k)

    def inside(self, points_xy, extra_m: float = 0.0) -> np.ndarray:
        """Which points (n, 2) are in the corridor (and within its length); `extra_m` widens it, for a point with a radius."""
        s, lat = self.coordinates(points_xy)
        return (np.abs(lat) <= self.half_width_m + extra_m) & (s >= -self.front_offset_m) & (s <= self.length_m)

    def distance_ahead(self, s) -> np.ndarray:
        """Distance the front of the car can still drive before reaching arc length s (never below 0)."""
        return np.maximum(np.asarray(s, dtype=float) - self.front_offset_m, 0.0)

    def centreline(self, n: int = 50, length_m: float | None = None) -> np.ndarray:
        """Points (n, 2) of the arc, for pictures and telemetry."""
        return point_on_path(np.linspace(0.0, self.length_m if length_m is None else length_m, n), self.k)


def make_corridor(state: EgoState, geometry: EgoGeometry = EgoGeometry(), cfg: EgoPathConfig = EgoPathConfig()) -> Corridor:
    return Corridor(k=curvature(state, geometry, cfg), half_width_m=geometry.width_m / 2.0 + cfg.lateral_margin_m,
                    length_m=cfg.length_m, front_offset_m=geometry.front_offset_m)