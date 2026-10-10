"""Synthetic LiDAR scenes for tests: a ray-cast of a terrain profile and boxes into a RangeImage, with the truth.

World frame: x forward, z up. The vehicle origin (rear axle, ground level) is the world origin, the car is pitched nose up
by `pitch` (rad) about it. The terrain only depends on x: piecewise linear, given as [(x_start, slope), ...] with
height 0 at x = 0. Boxes stand on the terrain at their centre.

render() returns (RangeImage, truth, geometry) with truth 0 = no hit, 1 = ground, 2 = object; scene.last_box_index tells
which box every cell hit.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from server.geometry.calibration import LidarCalibration, Mount
from server.perception.range_image import RangeImageBuilder

ELEVATIONS = [10, 7, 4, 2, 1, 0.5, 0, -0.5, -1, -1.5, -2, -2.5, -3, -3.5, -4, -4.5, -5, -6, -7, -8, -9, -10, -12, -14,
              -16, -18, -20]


def make_lidar(cols: int = 241, max_range: float = 100.0) -> LidarCalibration:
    az = np.linspace(60.0, -60.0, cols)
    return LidarCalibration(rows=len(ELEVATIONS), cols=cols, rate_hz=10.0, max_range_m=max_range,
                            mount=Mount(1.3, 0.0, 1.8, 0.0, 0.0, 0.0),
                            elevations_deg=np.array(ELEVATIONS, dtype=float), azimuths_deg=az)


class Scene:
    def __init__(self, terrain=((0.0, 0.0),)):
        self.terrain = list(terrain)
        self.boxes = []

    def ground_z(self, x: float) -> float:
        z = 0.0
        for i, (xs, s) in enumerate(self.terrain):
            xe = self.terrain[i + 1][0] if i + 1 < len(self.terrain) else math.inf
            if x <= xe:
                return z + s * (x - xs)
            z += s * (xe - xs)

    def add_box(self, x, y, length, width, height):
        z0 = self.ground_z(x)
        self.boxes.append((x - length / 2, x + length / 2, y - width / 2, y + width / 2, z0, z0 + height))
        return self

    def _terrain_hit(self, o, d):
        best = np.full(d.shape[:-1], np.inf)
        zs = 0.0
        for i, (xs, s) in enumerate(self.terrain):
            xe = self.terrain[i + 1][0] if i + 1 < len(self.terrain) else math.inf
            den = d[..., 2] - s * d[..., 0]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (zs + s * (o[0] - xs) - o[2]) / den
            x_hit = o[0] + t * d[..., 0]
            ok = (np.abs(den) > 1e-12) & (t > 0) & (x_hit >= xs) & (x_hit < xe)
            best = np.where(ok & (t < best), t, best)
            if np.isfinite(xe):
                zs += s * (xe - xs)
        return best

    def _box_hit(self, o, d, box):
        lo, hi = np.array([box[0], box[2], box[4]]), np.array([box[1], box[3], box[5]])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - o) / d, (hi - o) / d
        tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
        return np.where((tmax >= tmin) & (tmin > 0), tmin, np.inf)

    def render(self, lidar: LidarCalibration, pitch: float = 0.0, pose_pitch: float | None = None, dropout: float = 0.0,
               seed: int = 0):
        """pose_pitch: the pitch reported in the pose (default: the true one). dropout: fraction of cells set to no return."""
        b = RangeImageBuilder(lidar)
        s, c = math.sin(pitch), math.cos(pitch)
        rot = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])          # vehicle -> world, nose up by pitch
        o = rot @ b.geometry.origin
        d = b.geometry.dirs_vehicle @ rot.T
        t_ground = self._terrain_hit(o, d)
        t_box = np.full_like(t_ground, np.inf)
        self.last_box_index = np.full(t_ground.shape, -1, dtype=int)      # which box each cell hit (-1: none), for tests
        for i, box in enumerate(self.boxes):
            t_i = self._box_hit(o, d, box)
            self.last_box_index = np.where(t_i < t_box, i, self.last_box_index)
            t_box = np.minimum(t_box, t_i)
        t = np.minimum(t_ground, t_box)
        self.last_box_index = np.where(t_box < t_ground, self.last_box_index, -1)
        truth = np.where(t > lidar.max_range_m, 0, np.where(t_box < t_ground, 2, 1)).astype(np.uint8)
        raw = np.where(t > lidar.max_range_m, 65535, np.clip(np.round(t * 100), 1, 65534)).astype(np.uint16)
        if dropout > 0:
            raw[np.random.default_rng(seed).random(raw.shape) < dropout] = 0
        pp = pitch if pose_pitch is None else pose_pitch
        h = SimpleNamespace(t=0.0, seq=0, pose=(0.0, 0.0, 0.0, 0.0, pp, 0.0), speed=0.0, yaw_rate=0.0, steering_angle=0.0)
        img = b.convert(h, raw)
        truth = np.where(raw == 0, 0, truth)
        return img, truth, b.geometry