"""Measurement noise R of a cluster measurement (step 6.2, three measured points in 6.3): from the sensor noise and the range.

A LiDAR measurement is accurate along its line of sight and coarse across it, so R is not a circle. In the frame of the
line of sight (u from the sensor to the object, v = u turned 90 degrees to the left):

    sigma_along  (u)   range noise of the sensor, the quantisation of the range, a small part proportional to the range,
                       and for the corrected centre also the uncertainty of the size prior (the depth of the unseen
                       part of the object is a guess, `sigma_extent_m`)
    sigma_across (v)   a cell is range x azimuth step wide, and the true position lies anywhere inside it (uniform:
                       standard deviation cell / sqrt(12)); with the pose noise of the car (yaw error times range, position
                       error) and a floor

    R_world = Rot(bearing) diag(sigma_along^2, sigma_across^2) Rot(bearing)^T       (bearing in the world frame)

The across term grows with the range, which is what makes distant objects count for less in the filter and in the gate.
All numbers are parameters; the defaults come from the simulator configuration (range noise 0.03 m, azimuth step 0.2 degrees)
and are re-checked against the ground truth in step 6.9. The pose noise terms are zero by default (pose noise is a toggle
of the simulator); the pose-noise experiment of 6.10 sets them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from server.geometry.frames import line_of_sight_covariance
from server.tracking.measurement import WorldCluster


POINTS = ("nearest", "centre", "centroid")


@dataclass(frozen=True)
class MeasurementNoise:
    range_sigma_m: float = 0.03             # LiDAR range noise (lidar.rangeNoiseM of the configuration)
    range_quantum_m: float = 0.01           # the range is sent in whole centimetres
    range_relative_sigma: float = 0.0       # extra range error as a fraction of the range
    azimuth_step_rad: float = math.radians(0.2)
    sigma_extent_m: float = 0.7             # uncertainty of the unseen depth, added along the line of sight for the centre
    pose_position_sigma_m: float = 0.0      # pose noise of the car (position)
    pose_yaw_sigma_rad: float = 0.0         # pose noise of the car (yaw): moves a point at range r sideways by r sigma
    floor_m: float = 0.01                   # no standard deviation is smaller than this

    def sigmas(self, range_m: float, point: str = "nearest") -> tuple[float, float]:
        """(sigma along, sigma across) the line of sight at a range, for the measured point: 'nearest', 'centre' (corrected box centre) or 'centroid' (kept for the comparison of 6.3)."""
        if point not in POINTS:
            raise ValueError(f"point must be one of {POINTS}")
        along2 = (self.range_sigma_m ** 2 + self.range_quantum_m ** 2 / 12.0 + (self.range_relative_sigma * range_m) ** 2
                  + self.pose_position_sigma_m ** 2)
        if point != "nearest":                                                  # centre and centroid depend on the unseen depth
            along2 += self.sigma_extent_m ** 2
        cell = range_m * self.azimuth_step_rad
        across2 = cell ** 2 / 12.0 + (range_m * self.pose_yaw_sigma_rad) ** 2 + self.pose_position_sigma_m ** 2
        return max(math.sqrt(along2), self.floor_m), max(math.sqrt(across2), self.floor_m)

    def covariance(self, range_m: float, world_bearing: float, point: str = "nearest") -> np.ndarray:
        a, c = self.sigmas(range_m, point)
        return line_of_sight_covariance(world_bearing, a, c)


def measurement_of(w: WorldCluster, noise: MeasurementNoise = MeasurementNoise(), point: str = "nearest"):
    """(z, R, t) of a world cluster: the position of the chosen point, its covariance, and the capture time.
    The range of the covariance is the range of the chosen point from the sensor."""
    if point == "nearest":
        z = w.nearest_xy
    elif point == "centre":
        z = w.centre_xy
    elif point == "centroid":
        z = w.centroid_xy
    else:
        raise ValueError(f"point must be one of {POINTS}")
    rng = w.range_m if point == "nearest" else float(np.hypot(*(np.asarray(z) - w.sensor_xy)))
    return np.asarray(z, dtype=float), noise.covariance(rng, w.bearing, point), w.t