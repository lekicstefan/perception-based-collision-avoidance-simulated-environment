"""Clusters in the world frame (step 6.1): what the tracker receives from the perception.

A cluster is measured in the vehicle frame of the scan time (server/perception/cluster_geometry.py). The tracker works in the
world (start) frame, so every position is moved with the pose that was stamped on the same LiDAR message, that is the pose
at the capture time of the scan, not the pose at the moment the message is processed. A parked car then keeps the same world
position however fast the ego car drives, and track velocities are true object velocities.

Besides the positions this keeps what the later steps need and cannot recompute: the world position of the sensor at the
scan time and the world direction from the sensor to the cluster (for the covariance along and across the line of sight in
6.2 and for the bearing updates in 6.6), and the oriented box.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from server.geometry.frames import bearing_of, vehicle_to_world, wrap_angle
from server.perception.cluster_geometry import ClusterGeometry


@dataclass(frozen=True, eq=False)
class WorldCluster:
    t: float                         # capture time of the scan (s)
    pose: tuple                      # ego pose at that time (x, y, z, yaw, pitch, roll), start frame
    label: int                       # label of the cluster in its scan (only meaningful inside that scan)
    n_cells: int
    nearest_xy: np.ndarray           # (2,) world position of the point nearest to the sensor
    centre_xy: np.ndarray            # (2,) world position of the corrected box centre
    centroid_xy: np.ndarray          # (2,) world position of the centroid (for comparison)
    sensor_xy: np.ndarray            # (2,) world position of the sensor
    bearing: float                   # world direction sensor -> cluster, rad from the world x axis, positive to the left
    range_m: float                   # range from the sensor to the nearest point
    corners_xy: np.ndarray           # (4, 2) world corners of the vehicle-frame bounding box (turned with the yaw)
    width_m: float
    depth_m: float
    height_m: float
    top_m: float


def cluster_to_world(g: ClusterGeometry, pose, t: float, sensor_xy_vehicle) -> WorldCluster:
    """One cluster to the world frame with the pose of its scan. `sensor_xy_vehicle` is the sensor position in the vehicle
    frame (`LidarGeometry.origin[:2]`)."""
    x0, x1, y0, y1 = g.box_xy
    corners = vehicle_to_world(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), pose)
    sensor = vehicle_to_world(np.asarray(sensor_xy_vehicle, dtype=float)[:2], pose)
    return WorldCluster(
        t=float(t), pose=tuple(pose), label=g.label, n_cells=g.n_cells,
        nearest_xy=vehicle_to_world(g.nearest_point[:2], pose), centre_xy=vehicle_to_world(g.centre_xy, pose),
        centroid_xy=vehicle_to_world(g.centroid_xy, pose), sensor_xy=sensor,
        bearing=float(wrap_angle(g.bearing + pose[3])), range_m=float(g.nearest_range_m), corners_xy=corners,
        width_m=g.width_m, depth_m=g.depth_m, height_m=g.height_m, top_m=g.top_m)


def clusters_to_world(geoms: list, img, sensor_xy_vehicle) -> list:
    """All clusters of a scan, with the pose and time stamped on the scan (`img` is the RangeImage of the scan)."""
    return [cluster_to_world(g, img.pose, img.t, sensor_xy_vehicle) for g in geoms]


def bearing_check(w: WorldCluster) -> float:
    """Difference between the stored bearing and the direction from the sensor to the nearest point (rad, small); a
    consistency number for tests and for the world check tool."""
    return float(wrap_angle(w.bearing - bearing_of(w.nearest_xy, w.sensor_xy)))