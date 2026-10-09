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
from dataclasses import dataclass
from enum import Enum

import numpy as np

from server.perception.range_image import LidarGeometry, RangeImage


class GroundMethod(str, Enum):
    HEIGHT = "height"
    RANGE_IMAGE = "range_image"


@dataclass(frozen=True)
class GroundConfig:
    method: GroundMethod = GroundMethod.RANGE_IMAGE
    height_tol_m: float = 0.15       # HEIGHT: points lower than this above the ground plane are ground
    slope_tol_deg: float = 10.0      # RANGE_IMAGE: steepest slope still counted as ground


def _level_frame(points: np.ndarray, pitch: float):
    """Vehicle frame points -> (forward distance along the level plane, y, height above the ground plane)."""
    s, c = math.sin(pitch), math.cos(pitch)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return c * x - s * z, y, s * x + c * z


def height_filter_mask(img: RangeImage, geometry: LidarGeometry, height_tol_m: float = 0.15) -> np.ndarray:
    _, _, h = _level_frame(geometry.points_vehicle(img), img.pose[4])
    with np.errstate(invalid="ignore"):
        return img.valid & (h < height_tol_m)


def range_image_mask(img: RangeImage, geometry: LidarGeometry, slope_tol_deg: float = 10.0) -> np.ndarray:
    pitch = img.pose[4]
    xl, y, h = _level_frame(geometry.points_vehicle(img), pitch)
    rows, cols = img.shape
    tan_tol = math.tan(math.radians(slope_tol_deg))
    # start of every column's chain: the ground point under the sensor
    ox, oy, oz = geometry.origin
    s, c = math.sin(pitch), math.cos(pitch)
    last_x = np.full(cols, c * ox - s * oz)
    last_y = np.full(cols, float(oy))
    last_h = np.zeros(cols)
    ground = np.zeros((rows, cols), dtype=bool)
    valid = img.valid
    with np.errstate(invalid="ignore"):
        for r in range(rows - 1, -1, -1):              # row rows - 1 is the lowest beam
            run = np.hypot(xl[r] - last_x, y[r] - last_y)
            rise = np.abs(h[r] - last_h)
            g = valid[r] & (rise <= tan_tol * run)
            ground[r] = g
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
        return range_image_mask(img, self.geometry, self.config.slope_tol_deg)

    def candidates(self, img: RangeImage) -> np.ndarray:
        """Cells the segmentation may use: a return that is not ground."""
        return img.valid & ~self.ground_mask(img)