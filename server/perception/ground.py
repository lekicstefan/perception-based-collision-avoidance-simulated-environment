"""Ground removal (step 5.3).

Marks the cells of a RangeImage that hit the road, so the segmentation (step 5.4) only sees things standing on it.
Ground removal is needed on every road, not only on slopes: with the sensor 1.8 m up, a -1 degree beam meets flat ground at
about 100 m, so several near-horizon beams return ground out to the range limit. Those returns form long rows of similar
ranges that the beta criterion would cluster as objects and merge with anything standing on the road. The pipeline is
therefore never run without ground removal; there is deliberately no "none" method.

Two methods, selectable in GroundConfig (the ablation of step 11 compares them):

  HEIGHT       flat-ground height filter. A cell is ground when its point lies less than height_tol_m above the nominal
               ground plane. The plane passes through the vehicle origin (rear axle, ground level) and is tilted by the
               car's pitch from the pose. Simple and fast, but it assumes the road is one plane: ahead of a slope or
               a crest the road leaves the plane and is reported as obstacles (scenario 9).
  RANGE_IMAGE  range-image ground estimation, after Bogoslavskyi and Stachniss. Per column, from the lowest beam upwards:
               a cell is ground when the line from the last ground cell of that column to it is flatter than
               slope_tol_deg. The first comparison is against the point on the ground under the sensor. A road that
               bends up or down is followed as long as it stays within the slope tolerance, a wall or a car front is
               steeper and breaks the chain. Cells without a return are skipped (the chain continues over them).

Slopes are measured in a level frame: the vehicle frame turned by the car's pitch, so "level" means level in the world.
Pitch sign: positive = nose up (as the mount pitch of the calibration). Roll is not used.

Both functions return a boolean (rows, cols) array, True = ground. Only cells with a return can be ground.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np

from server.perception.range_image import CellState, LidarGeometry, RangeImage


class GroundMethod(str, Enum):
    HEIGHT = "height"
    RANGE_IMAGE = "range_image"


@dataclass(frozen=True)
class GroundConfig:
    method: GroundMethod = GroundMethod.RANGE_IMAGE
    height_tol_m: float = 0.15       # HEIGHT: points lower than this above the ground plane are ground
    slope_tol_deg: float = 10.0      # RANGE_IMAGE: steepest slope still counted as ground
    wall_slope_deg: float = 45.0     # RANGE_IMAGE: a step this steep from the previous return below is never ground
    row_deviation_m: float = 0.25    # RANGE_IMAGE: ground this much higher than the ground beside it, in the same row, is not
    row_window: int = 2              # ... "beside" = this many columns to each side (0 or row_deviation_m 0 switches it off)
    isolated_run_max: int = 4        # RANGE_IMAGE: a run of at most this many returns with max range on both sides is an object


def level_frame(points: np.ndarray, pitch: float):
    """Vehicle frame points -> (forward distance along the level plane, y, height above the ground plane)."""
    s, c = math.sin(pitch), math.cos(pitch)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return c * x - s * z, y, s * x + c * z


def height_filter_mask(img: RangeImage, geometry: LidarGeometry, height_tol_m: float = 0.15) -> np.ndarray:
    _, _, h = level_frame(geometry.points_vehicle(img), img.pose[4])
    with np.errstate(invalid="ignore"):
        return img.valid & (h < height_tol_m)


def _neighbour_ground_height(row_h: np.ndarray, row_ground: np.ndarray, k: int) -> tuple:
    """Median height of the ground cells within k columns of every cell of one row (the cell itself left out), and how many."""
    n = len(row_h)
    vals = np.where(row_ground, row_h, np.nan)
    padded = np.concatenate([np.full(k, np.nan), vals, np.full(k, np.nan)])
    stack = np.stack([padded[i:i + n] for i in range(2 * k + 1) if i != k])             # (2k, n)
    count = np.count_nonzero(~np.isnan(stack), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)                      # all-NaN columns give NaN, as wanted
        return np.nanmedian(stack, axis=0), count


def _isolated_runs(valid_row: np.ndarray, max_range_row: np.ndarray, run_max: int) -> np.ndarray:
    """Cells of narrow runs of returns that have a ray that reached max range right next to them on both sides.

    Flat ground gives the same range along a row, so a row either returns ground across the whole view or not at all; a
    few returns in a row that otherwise passes by are an object (a person far away, where the beams are metres apart)."""
    out = np.zeros(len(valid_row), dtype=bool)
    v = np.concatenate([[False], valid_row, [False]]).astype(np.int8)
    starts, ends = np.flatnonzero(np.diff(v) == 1), np.flatnonzero(np.diff(v) == -1) - 1       # inclusive cell indices
    for a, b in zip(starts, ends):
        if b - a + 1 <= run_max and a > 0 and b < len(valid_row) - 1 and max_range_row[a - 1] and max_range_row[b + 1]:
            out[a:b + 1] = True
    return out


def range_image_mask(img: RangeImage, geometry: LidarGeometry, slope_tol_deg: float = 10.0, wall_slope_deg: float = 45.0,
                     row_deviation_m: float = 0.25, row_window: int = 2, isolated_run_max: int = 4) -> np.ndarray:
    pitch = img.pose[4]
    xl, y, h = level_frame(geometry.points_vehicle(img), pitch)
    rows, cols = img.shape
    tan_tol = math.tan(math.radians(slope_tol_deg))
    tan_wall = math.tan(math.radians(wall_slope_deg))
    # start of every column's chain: the ground point under the sensor
    ox, oy, oz = geometry.origin
    s, c = math.sin(pitch), math.cos(pitch)
    last_x = np.full(cols, c * ox - s * oz)
    last_y = np.full(cols, float(oy))
    last_h = np.zeros(cols)
    ground = np.zeros((rows, cols), dtype=bool)
    prev_x, prev_y, prev_h = np.full(cols, np.nan), np.full(cols, np.nan), np.full(cols, np.nan)   # previous return below
    valid = img.valid
    with np.errstate(invalid="ignore"):
        for r in range(rows - 1, -1, -1):              # row rows - 1 is the lowest beam
            run = np.hypot(xl[r] - last_x, y[r] - last_y)
            rise = np.abs(h[r] - last_h)
            g = valid[r] & (rise <= tan_tol * run)
            # The lowest cells of a thin object far away are a gentle rise from the last ground cell (the beams are metres
            # apart there). Two more tests catch them: the cell above one object cell is a near-vertical step from it, and
            # the cell is higher than the ground in the neighbouring columns of the same row.
            g &= (np.abs(h[r] - prev_h) <= tan_wall * np.hypot(xl[r] - prev_x, y[r] - prev_y)) | np.isnan(prev_h)
            if isolated_run_max > 0:
                g &= ~_isolated_runs(valid[r], img.state[r] == CellState.MAX_RANGE, isolated_run_max)
            if row_deviation_m > 0 and row_window > 0:
                med, count = _neighbour_ground_height(h[r], g, row_window)
                g &= ~((count >= 2) & (h[r] - med > row_deviation_m))
            ground[r] = g
            prev_x, prev_y, prev_h = (np.where(valid[r], xl[r], prev_x), np.where(valid[r], y[r], prev_y),
                                      np.where(valid[r], h[r], prev_h))
            last_x = np.where(g, xl[r], last_x)
            last_y = np.where(g, y[r], last_y)
            last_h = np.where(g, h[r], last_h)
    return ground


class GroundRemover:
    def __init__(self, geometry: LidarGeometry, config: GroundConfig | None = None):
        self.geometry = geometry
        self.config = config or GroundConfig()
        self.method = GroundMethod(self.config.method)       # raises ValueError for anything else, including "none"

    def ground_mask(self, img: RangeImage) -> np.ndarray:
        if self.method is GroundMethod.HEIGHT:
            return height_filter_mask(img, self.geometry, self.config.height_tol_m)
        c = self.config
        return range_image_mask(img, self.geometry, c.slope_tol_deg, c.wall_slope_deg, c.row_deviation_m, c.row_window,
                                c.isolated_run_max)

    def candidates(self, img: RangeImage) -> np.ndarray:
        """Cells the segmentation may use: a return that is not ground."""
        return img.valid & ~self.ground_mask(img)