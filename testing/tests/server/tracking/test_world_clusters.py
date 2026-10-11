"""Clusters in the world frame (step 6.1), on synthetic scenes."""
import math

import numpy as np
import pytest

from server.geometry.calibration import Calibration, EgoCalibration
from server.geometry.frames import vehicle_to_world
from server.perception.pipeline import PerceptionPipeline
from server.tracking.measurement import bearing_check, clusters_to_world
from testing.support.synthetic_camera import make_camera
from testing.support.synthetic_scene import Scene, make_lidar

LIDAR = make_lidar()
CALIB = Calibration("t", EgoCalibration(2.7, 4.4, 1.8, 1.5, 0.9, 0.8), make_camera(), LIDAR, 100.0)


def world_clusters(scene, pose):
    """What the sensor sees of the scene from a car at `pose`, as world clusters (the scene is given in the vehicle frame)."""
    img = scene.render(LIDAR)[0]
    img = type(img)(img.t, img.seq, tuple(pose), img.speed, img.yaw_rate, img.steering_angle, img.range_m, img.state)
    pipe = PerceptionPipeline(CALIB)
    result = pipe.process(img)
    return clusters_to_world(result.clusters, img, pipe.builder.geometry.origin[:2]), result


def largest(world, result):
    """The biggest cluster of the scan (a far object may leave a small fragment next to it) in both frames."""
    i = max(range(len(world)), key=lambda k: world[k].n_cells)
    return world[i], result.clusters[i]


def test_the_world_position_of_a_cluster_follows_the_pose_of_its_scan():
    scene = Scene().add_box(25, 3, 4.5, 1.8, 1.5)
    a, _ = largest(*world_clusters(scene, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    b, _ = largest(*world_clusters(scene, (10.0, 5.0, 0.0, 0.0, 0.0, 0.0)))
    assert b.nearest_xy - a.nearest_xy == pytest.approx([10.0, 5.0], abs=1e-9)
    assert b.centre_xy - a.centre_xy == pytest.approx([10.0, 5.0], abs=1e-9)
    assert b.range_m == pytest.approx(a.range_m)                                  # a range does not depend on the pose


def test_a_parked_car_is_at_the_same_world_position_from_two_different_poses():
    """The same car in the world seen from two poses: the scene is built in the vehicle frame of each pose."""
    car_world, yaw = np.array([30.0, 4.0]), 0.5
    pose_a = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pose_b = (8.0, 1.0, 0.0, yaw, 0.0, 0.0)
    out = []
    for pose in (pose_a, pose_b):
        local = world_to_local(car_world, pose)
        scene = Scene().add_box(float(local[0]), float(local[1]), 0.6, 0.5, 1.7)        # a person-sized post, yaw-free
        w, _ = largest(*world_clusters(scene, pose))
        out.append(w.centre_xy)
    assert np.linalg.norm(out[0] - out[1]) < 0.6                                      # the same point, up to the prior's bias
    assert np.linalg.norm(out[0] - car_world) < 1.0


def world_to_local(xy, pose):
    from server.geometry.frames import world_to_vehicle
    return world_to_vehicle(xy, pose)


def test_the_world_bearing_and_the_sensor_position_use_the_pose():
    scene = Scene().add_box(25, 6, 0.6, 0.5, 1.7)
    pose = (4.0, 2.0, 0.0, math.pi / 2, 0.0, 0.0)                                       # facing left
    w, g = largest(*world_clusters(scene, pose))
    assert w.sensor_xy == pytest.approx(vehicle_to_world(np.array([1.3, 0.0]), pose))
    assert w.bearing == pytest.approx(g.bearing + math.pi / 2, abs=1e-9)
    assert abs(bearing_check(w)) < 0.05                                                 # towards the nearest point, nearly
    assert w.bearing > math.pi / 2                                                      # the object is on the left of the car


def test_the_world_box_corners_are_the_vehicle_box_turned_with_the_yaw():
    scene = Scene().add_box(25, 0, 4.5, 1.8, 1.5)
    pose = (0.0, 0.0, 0.0, math.pi / 2, 0.0, 0.0)
    w, g = largest(*world_clusters(scene, pose))
    x0, x1, y0, y1 = g.box_xy
    assert w.corners_xy.shape == (4, 2)
    assert w.corners_xy[:, 0].min() == pytest.approx(-y1) and w.corners_xy[:, 0].max() == pytest.approx(-y0)
    assert w.corners_xy[:, 1].min() == pytest.approx(x0) and w.corners_xy[:, 1].max() == pytest.approx(x1)