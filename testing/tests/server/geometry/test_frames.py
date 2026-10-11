"""Frames and conventions (step 6.1), with known poses and the left-handed to right-handed conversion."""
import math

import numpy as np
import pytest

from server.geometry.frames import (Pose, bearing_of, line_of_sight_covariance, rotate_covariance, rotate_to_vehicle,
                                    rotate_to_world, to_start_frame, unity_to_processor, unity_vector_to_processor,
                                    vehicle_to_world, world_to_vehicle, wrap_angle)

HALF_PI = math.pi / 2


def test_wrap_angle_is_in_minus_pi_exclusive_to_pi_inclusive():
    assert wrap_angle(math.pi) == pytest.approx(math.pi)
    assert wrap_angle(-math.pi) == pytest.approx(math.pi)
    assert wrap_angle(3 * math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
    assert wrap_angle(-0.3) == pytest.approx(-0.3)
    assert wrap_angle(np.array([0.0, 2 * math.pi, -2 * math.pi + 0.5])) == pytest.approx([0.0, 0.0, 0.5])


# ---- vehicle <-> world with known poses ---------------------------------------------------------------------------------
def test_vehicle_to_world_with_known_poses():
    assert vehicle_to_world((10.0, 0.0), (5.0, 7.0, 0.0, 0.0, 0, 0)) == pytest.approx([15.0, 7.0])
    assert vehicle_to_world((10.0, 0.0), (5.0, 7.0, 0.0, HALF_PI, 0, 0)) == pytest.approx([5.0, 17.0])    # facing +y (left)
    assert vehicle_to_world((0.0, 2.0), (0.0, 0.0, 0.0, HALF_PI, 0, 0)) == pytest.approx([-2.0, 0.0])      # 2 m to its left
    assert vehicle_to_world((10.0, 0.0), (0.0, 0.0, 0.0, -HALF_PI, 0, 0)) == pytest.approx([0.0, -10.0])   # facing right
    assert vehicle_to_world((10.0, 0.0), (0.0, 0.0, 0.0, math.pi, 0, 0)) == pytest.approx([-10.0, 0.0])    # facing back


def test_a_point_ahead_and_to_the_left_stays_ahead_and_to_the_left_whatever_the_heading():
    for yaw in np.linspace(-3.0, 3.0, 13):
        pose = (4.0, -2.0, 0.0, float(yaw), 0, 0)
        w = vehicle_to_world((10.0, 2.0), pose)
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        left = np.array([-math.sin(yaw), math.cos(yaw)])
        d = w - np.array([4.0, -2.0])
        assert d @ forward == pytest.approx(10.0) and d @ left == pytest.approx(2.0)


def test_world_to_vehicle_is_the_inverse_and_works_on_arrays():
    pose = (3.0, -4.0, 0.0, 0.7, 0.01, 0.02)
    pts = np.array([[1.0, 2.0], [-5.0, 0.5], [30.0, -8.0]])
    assert world_to_vehicle(vehicle_to_world(pts, pose), pose) == pytest.approx(pts)
    assert vehicle_to_world(pts, pose).shape == (3, 2)
    assert vehicle_to_world(np.zeros((2, 3, 2)), pose).shape == (2, 3, 2)


def test_vectors_are_rotated_but_not_shifted():
    pose = (100.0, 50.0, 0.0, HALF_PI, 0, 0)
    assert rotate_to_world((3.0, 0.0), pose) == pytest.approx([0.0, 3.0])         # 3 m/s ahead while facing left
    assert rotate_to_vehicle((0.0, 3.0), pose) == pytest.approx([3.0, 0.0])


def test_a_parked_car_keeps_its_world_position_while_the_ego_drives_past():
    car_world = np.array([40.0, 3.0])
    for x_ego in (0.0, 10.0, 25.0, 38.0):
        pose = (x_ego, 0.0, 0.0, 0.0, 0, 0)
        measured = world_to_vehicle(car_world, pose)                              # what the sensor would see
        assert vehicle_to_world(measured, pose) == pytest.approx(car_world)
        assert measured == pytest.approx([40.0 - x_ego, 3.0])


def test_bearing_is_positive_to_the_left():
    assert bearing_of((10.0, 10.0)) == pytest.approx(math.pi / 4)
    assert bearing_of((10.0, -10.0)) == pytest.approx(-math.pi / 4)
    assert bearing_of((5.0, 3.0), origin=(5.0, 1.0)) == pytest.approx(HALF_PI)


# ---- covariance ---------------------------------------------------------------------------------------------------------
def test_covariance_turns_with_the_yaw_and_keeps_its_eigenvalues():
    c = np.diag([4.0, 1.0])
    turned = rotate_covariance(c, HALF_PI)
    assert turned == pytest.approx(np.diag([1.0, 4.0]), abs=1e-12)
    r = rotate_covariance(c, 0.6)
    assert np.sort(np.linalg.eigvalsh(r)) == pytest.approx([1.0, 4.0])
    assert r == pytest.approx(r.T)


def test_line_of_sight_covariance_is_long_along_the_line_of_sight():
    c = line_of_sight_covariance(HALF_PI, sigma_along=0.5, sigma_across=0.1)         # looking along +y
    assert c[1, 1] == pytest.approx(0.25) and c[0, 0] == pytest.approx(0.01)
    c45 = line_of_sight_covariance(math.pi / 4, 0.5, 0.1)
    assert c45[0, 1] > 0                                                              # elongated along the diagonal


# ---- the left-handed to right-handed conversion (the simulator boundary) ----------------------------------------------------
def test_unity_position_and_yaw_become_a_right_handed_pose():
    # Unity: x right, y up, z forward. A car at x = 3 (right), 10 ahead, 1 up, heading 90 deg clockwise (to Unity +x).
    p = unity_to_processor((3.0, 1.0, 10.0), (5.0, 90.0, 2.0))
    assert tuple(p[:3]) == pytest.approx((10.0, -3.0, 1.0))                            # forward 10, 3 to the RIGHT = y -3
    assert p.yaw == pytest.approx(-HALF_PI)                                            # turned right = negative yaw
    assert p.pitch == pytest.approx(-math.radians(5.0))                                # Unity positive pitch is nose down
    assert p.roll == pytest.approx(-math.radians(2.0))


def test_an_object_on_the_left_in_unity_has_positive_y():
    left_unity = unity_to_processor((-2.0, 0.0, 10.0), (0.0, 0.0, 0.0))                # Unity x = -2 is to the left
    assert left_unity.y == pytest.approx(2.0) and left_unity.x == pytest.approx(10.0)


def test_the_heading_vector_agrees_in_both_conventions():
    for yaw_deg in (0.0, 30.0, 90.0, 135.0, 270.0):
        forward_unity = (math.sin(math.radians(yaw_deg)), math.cos(math.radians(yaw_deg)))     # (x right, z forward)
        v = unity_vector_to_processor((forward_unity[0], 0.0, forward_unity[1]))
        p = unity_to_processor((0.0, 0.0, 0.0), (0.0, yaw_deg, 0.0))
        assert v == pytest.approx([math.cos(p.yaw), math.sin(p.yaw)], abs=1e-12)


def test_unity_euler_angles_above_180_degrees_are_wrapped():
    p = unity_to_processor((0, 0, 0), (350.0, 270.0, 355.0))                           # pitch -10, yaw -90, roll -5 in Unity terms
    assert p.pitch == pytest.approx(math.radians(10.0)) and p.roll == pytest.approx(math.radians(5.0))
    assert p.yaw == pytest.approx(HALF_PI)                                             # Unity 270 = 90 to the left


def test_the_start_pose_becomes_the_origin_and_the_run_is_expressed_in_its_frame():
    start = unity_to_processor((5.0, 0.0, 5.0), (0.0, 90.0, 0.0))                      # at (5, 5) heading Unity +x
    now = unity_to_processor((15.0, 0.0, 5.0), (0.0, 90.0, 0.0))                       # 10 m further along its heading
    assert tuple(to_start_frame(start, start)[:4]) == pytest.approx((0.0, 0.0, 0.0, 0.0), abs=1e-12)
    rel = to_start_frame(now, start)
    assert (rel.x, rel.y, rel.yaw) == pytest.approx((10.0, 0.0, 0.0), abs=1e-12)
    # 3 m to the left of the start heading is positive y in the start frame (heading Unity +x: its left is Unity +z)
    left_of_start = unity_to_processor((5.0, 0.0, 8.0), (0.0, 90.0, 0.0))
    assert to_start_frame(left_of_start, start).y == pytest.approx(3.0)


def test_pose_is_a_plain_tuple_compatible_with_the_message_header():
    p = Pose(1.0, 2.0, 3.0, 0.4, 0.05, 0.06)
    assert p[3] == 0.4 and tuple(p) == (1.0, 2.0, 3.0, 0.4, 0.05, 0.06)
    assert vehicle_to_world((1.0, 0.0), tuple(p)) == pytest.approx(vehicle_to_world((1.0, 0.0), p))


# ---- the C# conversion of the simulator (PoseEstimator.cs), ported line by line --------------------------------------------
def cs_yaw_to_start(unity_yaw_deg, psi0):
    """PoseEstimator.YawToStart: Unity yaw in degrees -> start-frame yaw in radians."""
    psi = (90.0 - unity_yaw_deg) * math.pi / 180.0
    return math.remainder(psi - psi0, 2.0 * math.pi)


def cs_to_start_frame(x0, z0, psi0, x, z, psi):
    """PoseEstimator.ToStartFrame: psi is counter-clockwise from Unity +x seen from above (x right, z up the page)."""
    dx, dz = x - x0, z - z0
    c, s = math.cos(psi0), math.sin(psi0)
    return dx * c + dz * s, -dx * s + dz * c, math.remainder(psi - psi0, 2.0 * math.pi)


def test_the_python_conversion_agrees_with_the_csharp_pose_estimator():
    rng = np.random.default_rng(3)
    for _ in range(200):
        x0, z0, x, z = rng.uniform(-80.0, 80.0, 4)
        yaw0, yaw = rng.uniform(0.0, 360.0, 2)                                       # Unity yaw, degrees
        psi0, psi = (90.0 - yaw0) * math.pi / 180.0, (90.0 - yaw) * math.pi / 180.0
        cx, cy, cyaw = cs_to_start_frame(x0, z0, psi0, x, z, psi)
        mine = to_start_frame(unity_to_processor((x, 0.0, z), (0.0, yaw, 0.0)),
                              unity_to_processor((x0, 0.0, z0), (0.0, yaw0, 0.0)))
        assert (mine.x, mine.y) == pytest.approx((cx, cy), abs=1e-9)
        assert wrap_angle(mine.yaw - cyaw) == pytest.approx(0.0, abs=1e-9)
        assert cs_yaw_to_start(yaw, psi0) == pytest.approx(cyaw, abs=1e-9) or \
            wrap_angle(cs_yaw_to_start(yaw, psi0) - cyaw) == pytest.approx(0.0, abs=1e-9)


def test_the_car_starts_at_the_origin_facing_forward_and_a_left_turn_is_positive():
    x0, z0, psi0 = 12.0, -7.0, (90.0 - 35.0) * math.pi / 180.0                        # Unity yaw 35 degrees at the start
    assert cs_to_start_frame(x0, z0, psi0, x0, z0, psi0) == pytest.approx((0.0, 0.0, 0.0))
    ahead = (x0 + 10.0 * math.cos(psi0), z0 + 10.0 * math.sin(psi0))                   # 10 m along the start heading
    assert cs_to_start_frame(x0, z0, psi0, ahead[0], ahead[1], psi0)[:2] == pytest.approx((10.0, 0.0))
    left = (x0 - 3.0 * math.sin(psi0), z0 + 3.0 * math.cos(psi0))                       # 3 m to its left (turned 90 deg CCW)
    assert cs_to_start_frame(x0, z0, psi0, left[0], left[1], psi0)[:2] == pytest.approx((0.0, 3.0))
    assert cs_yaw_to_start(35.0 - 20.0, psi0) > 0                                       # Unity yaw decreased = turned left