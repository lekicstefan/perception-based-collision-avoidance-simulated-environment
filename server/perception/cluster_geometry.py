"""Cluster geometry (step 5.5): what the tracker and the risk logic measure of a cluster.

For every cluster of the segmentation: its 3D points, the nearest point, a ground-plane bounding box, the visible
extent seen from the sensor, the height, and a corrected box centre.

Why a corrected centre. A LiDAR sees only the near surface of an object. The centroid of the visible points therefore
moves when the viewing angle changes (more of a car's side comes into view as it passes, the centroid slides along it),
which the tracker would read as velocity. Two quantities do not slide that way and are the preferred measurement points:

  * the nearest point (also the conservative quantity for collision risk), and
  * the corrected box centre: the centre of the visible extent, pushed back along the line of sight by a size prior.
        u = unit vector from the sensor to the cluster (horizontal), v = u turned 90 degrees to the left
        lateral centre = middle of the visible extent along v
        depth of the object along u = max(visible depth, size_prior_m)
        centre = sensor + u * (nearest depth + depth / 2) + v * lateral centre
    The prior stands in for the part of the object that cannot be seen. It is one number for every object (no object is
    treated differently by appearance or size) and it is a bias that is constant while the object is seen the same way,
    which is what the tracker needs. It is tuned against the ground truth in the evaluation (step 5.7).

All positions are in the vehicle frame of the scan time (x forward, y left, origin at the rear axle on the ground).
world_xy() turns a point into the start-frame (world) coordinates with the pose of the scan. Heights are above the nominal
ground plane (tilted by the car's pitch, as in ground.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from server.perception.ground import level_frame
from server.perception.range_image import LidarGeometry, RangeImage
from server.perception.segmentation import Cluster, Segmentation


@dataclass(frozen=True, eq=False)
class ClusterGeometry:
    label: int
    n_cells: int
    points: np.ndarray               # (n, 3) vehicle frame
    nearest_point: np.ndarray        # (3,) the point with the smallest range from the sensor
    nearest_range_m: float           # that range, from the sensor
    centre_xy: np.ndarray            # (2,) corrected box centre
    centroid_xy: np.ndarray          # (2,) mean of the visible points (for comparison only, do not track this)
    box_xy: tuple                    # (x_min, x_max, y_min, y_max) axis-aligned in the vehicle frame
    bearing: float                   # rad, direction of the cluster seen from the sensor (positive to the left)
    width_m: float                   # visible extent across the line of sight (cell centres, plus one cell width)
    depth_m: float                   # visible extent along the line of sight
    height_m: float                  # vertical extent of the visible points
    top_m: float                     # height of the highest point above the ground plane


def world_xy(xy, pose) -> np.ndarray:
    """Vehicle-frame ground position -> start-frame position. pose = (x, y, z, yaw, pitch, roll), yaw positive to the left."""
    c, s = math.cos(pose[3]), math.sin(pose[3])
    return np.array([pose[0] + c * xy[0] - s * xy[1], pose[1] + s * xy[0] + c * xy[1]])


class GeometryExtractor:
    def __init__(self, geometry: LidarGeometry, size_prior_m: float = 1.0):
        self.geometry = geometry
        self.size_prior_m = size_prior_m
        self.az_step = float(geometry.alpha_h.mean())                 # rad between neighbouring columns

    def extract_all(self, img: RangeImage, seg: Segmentation) -> list:
        pts = self.geometry.points_vehicle(img)
        _, _, h = level_frame(pts, img.pose[4])
        return [self._one(c, pts, h, img.range_m) for c in seg.clusters]

    def extract(self, img: RangeImage, cluster: Cluster) -> ClusterGeometry:
        pts = self.geometry.points_vehicle(img)
        _, _, h = level_frame(pts, img.pose[4])
        return self._one(cluster, pts, h, img.range_m)

    def _one(self, c: Cluster, pts, h, range_m) -> ClusterGeometry:
        p = pts[c.rows, c.cols]                                   # (n, 3)
        hh = h[c.rows, c.cols]
        origin = self.geometry.origin[:2]
        i = int(np.argmin(range_m[c.rows, c.cols]))
        centroid = p[:, :2].mean(axis=0)
        to = centroid - origin
        u = to / np.linalg.norm(to)
        v = np.array([-u[1], u[0]])
        pu, pv = (p[:, :2] - origin) @ u, (p[:, :2] - origin) @ v
        depth = float(pu.max() - pu.min())
        centre = origin + u * (pu.min() + max(depth, self.size_prior_m) / 2.0) + v * (pv.min() + pv.max()) / 2.0
        return ClusterGeometry(
            label=c.label, n_cells=len(p), points=p, nearest_point=p[i], nearest_range_m=float(range_m[c.rows[i], c.cols[i]]),
            centre_xy=centre, centroid_xy=centroid,
            box_xy=(float(p[:, 0].min()), float(p[:, 0].max()), float(p[:, 1].min()), float(p[:, 1].max())),
            bearing=math.atan2(to[1], to[0]), depth_m=depth,
            width_m=float(pv.max() - pv.min()) + float(np.median(range_m[c.rows, c.cols])) * self.az_step,      # + one cell
            height_m=float(hh.max() - hh.min()), top_m=float(hh.max()))