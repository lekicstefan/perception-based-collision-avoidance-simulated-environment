from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from server.geometry.frames import vehicle_to_world
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
    """Vehicle-frame ground position -> start-frame position (kept under this name; the one implementation is in
    server/geometry/frames.py). pose = (x, y, z, yaw, pitch, roll), yaw positive to the left."""
    return vehicle_to_world(xy, pose)


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