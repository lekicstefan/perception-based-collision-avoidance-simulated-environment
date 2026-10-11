"""Bearing updates of the filter, bearing measurements from the camera, and the tracker's camera input (step 6.6)."""
import math

import numpy as np
import pytest

from server.tracking.bearing import (BearingConfig, camera_origin_vehicle, camera_xy_world, measure_bearing, pixel_bearing_vehicle,
                                     prepare_frame)
from server.tracking.kalman import ConstantVelocityFilter
from server.tracking.track import TrackState
from server.tracking.tracker import Tracker, TrackerConfig
from testing.support.synthetic_camera import make_camera, render
from testing.tests.server.tracking.test_tracker import DT, wc

CAM = make_camera()                                           # 640 x 360, 90 degrees, mounted at (1.5, 0, 1.4)
POSE0 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
CAM_XY = (1.5, 0.0)
BOX = (25.0, 3.0, 4.5, 1.8, 1.5)                              # x, y, length, width, height


def flt(x=30.0, y=0.0, t=0.0):
    return ConstantVelocityFilter(np.array([x, y, 0.0, 0.0]), np.diag([4.0, 4.0, 1.0, 1.0]), t)


# ---- the extended Kalman update --------------------------------------------------------------------------------------
def test_the_measurement_jacobian_matches_finite_differences():
    x = np.array([30.0, 5.0, 0.0, 0.0])
    _, h = ConstantVelocityFilter._bearing_model(x, CAM_XY)
    eps = 1e-6
    for i in range(2):
        d = np.zeros(4)
        d[i] = eps
        num = (ConstantVelocityFilter._bearing_model(x + d, CAM_XY)[0] - ConstantVelocityFilter._bearing_model(x - d, CAM_XY)[0]) / (2 * eps)
        assert h[0, i] == pytest.approx(num, rel=1e-5)
    assert h[0, 2] == 0.0 and h[0, 3] == 0.0


def test_a_bearing_corrects_the_position_across_the_line_of_sight_and_not_along_it():
    f = flt()
    r = f.update_bearing(math.atan2(1.0, 28.5), 0.002, CAM_XY, 0.0)
    assert r.applied and r.reason == "same_time"
    assert f.position[1] == pytest.approx(1.0, abs=0.15)             # pulled across to the measured direction
    assert f.position[0] == pytest.approx(30.0, abs=1e-6)             # the range is untouched
    assert f.P[1, 1] < 0.1 and f.P[0, 0] == pytest.approx(4.0, abs=1e-6)       # only the cross uncertainty shrank
    assert np.allclose(f.P, f.P.T) and np.all(np.linalg.eigvalsh(f.P) > 0)


def test_the_innovation_is_a_wrapped_angle():
    f = flt(-30.0, 0.1)                                              # straight behind the camera: bearing near +pi
    f.update_bearing(-math.pi + 0.001, 0.002, (0.0, 0.0), 0.0)        # the same direction, written on the other side of the cut
    assert abs(f.position[1] - 0.1) < 0.5 and abs(f.position[0] + 30.0) < 0.05


def test_the_gate_distance_is_small_for_the_true_direction_and_large_for_a_wrong_one():
    f = flt()
    assert f.gate_distance2_bearing(0.0, 0.002, CAM_XY, 0.0) < 1e-9
    assert f.gate_distance2_bearing(0.3, 0.002, CAM_XY, 0.0) > 9.0                  # well outside a 3 sigma gate
    x, p, t = f.x.copy(), f.P.copy(), f.t
    assert f.x == pytest.approx(x) and f.P == pytest.approx(p) and f.t == t           # the gate does not change the track


def test_a_late_bearing_is_applied_without_moving_the_track_time_and_a_very_old_one_is_rejected():
    f = flt(t=1.0)
    r = f.update_bearing(math.atan2(0.5, 28.5), 0.002, CAM_XY, 0.9)
    assert r.applied and r.reason == "out_of_order" and f.t == 1.0
    assert not f.update_bearing(0.0, 0.002, CAM_XY, 0.0).applied


def test_repeated_bearings_keep_a_consistent_covariance():
    f = flt()
    rng = np.random.default_rng(3)
    nis = []
    for k in range(200):
        theta = math.atan2(0.0, 28.5) + rng.normal(0.0, 0.004)
        nis.append(f.update_bearing(theta, 0.004, CAM_XY, 0.0).nis)
    assert np.mean(nis[:20]) < 3.0 and np.all(np.linalg.eigvalsh(f.P) > 0)             # chi-square with one degree of freedom: mean 1


# ---- geometry of the camera -------------------------------------------------------------------------------------------
def test_the_camera_origin_is_its_mount_and_the_pixel_bearing_follows_the_sign_convention():
    assert camera_origin_vehicle(CAM) == pytest.approx([1.5, 0.0, 1.4])
    assert pixel_bearing_vehicle(CAM, CAM.cx) == pytest.approx(0.0, abs=1e-12)
    assert pixel_bearing_vehicle(CAM, CAM.cx - 50.0) > 0.0 > pixel_bearing_vehicle(CAM, CAM.cx + 50.0)   # image left = vehicle left
    u, v, _ = CAM.project(np.array([[25.0, 3.0, 0.8]]))
    assert pixel_bearing_vehicle(CAM, float(u[0]), float(v[0])) == pytest.approx(math.atan2(3.0, 23.5), abs=1e-6)


def test_the_camera_position_in_the_world_follows_the_pose():
    assert camera_xy_world(CAM, (10.0, 5.0, 0.0, math.pi / 2, 0.0, 0.0)) == pytest.approx((10.0, 6.5))     # forward is world +y


# ---- a bearing from an image -----------------------------------------------------------------------------------------
def tracker_with_box(n=3, tracker=None):
    tr = tracker or Tracker()
    for k in range(n):
        tr.step([wc(BOX[0], BOX[1], k * DT, width=BOX[3])], k * DT)
    return tr, render(CAM, [BOX])


def test_the_bearing_of_a_box_in_the_image_matches_its_true_direction():
    tr, rgb = tracker_with_box()
    m = measure_bearing(tr.tracks[0], prepare_frame(rgb), CAM, POSE0, 2 * DT)
    assert m.reason == "ok" and m.cam_xy == pytest.approx(CAM_XY)
    assert m.theta == pytest.approx(math.atan2(3.0, 23.5), abs=0.015)
    assert m.sigma == pytest.approx(math.hypot(2.0 / CAM.fx, 0.002))


def test_the_bearing_in_the_world_frame_adds_the_yaw_of_the_car():
    tr, _ = tracker_with_box()
    yaw = 0.2
    pose = (0.0, 0.0, 0.0, yaw, 0.0, 0.0)
    # the track is at (25, 3) in the world; with the car turned left by 0.2 rad it is further to the right in the image
    c, s = math.cos(yaw), math.sin(yaw)
    pv = np.array([c * 25 + s * 3, -s * 25 + c * 3])                  # the box as the turned car sees it
    rgb = render(CAM, [(pv[0], pv[1], BOX[2], BOX[3], BOX[4])])
    m = measure_bearing(tr.tracks[0], prepare_frame(rgb), CAM, pose, 2 * DT)
    cam = camera_xy_world(CAM, pose)
    assert m.reason == "ok" and m.theta == pytest.approx(math.atan2(3.0 - cam[1], 25.0 - cam[0]), abs=0.02)


def test_no_measurement_where_the_image_shows_nothing_or_nothing_useful():
    tr, _ = tracker_with_box()
    tk = tr.tracks[0]
    assert measure_bearing(tk, prepare_frame(render(CAM, [])), CAM, POSE0, 0.2).reason in ("no_edge", "low_contrast")     # empty road
    assert measure_bearing(tk, prepare_frame(render(CAM, [], flat=True)), CAM, POSE0, 0.2).reason == "low_contrast"
    behind = Tracker()
    behind.step([wc(-10.0, 0.0, 0.0)], 0.0)
    assert measure_bearing(behind.tracks[0], prepare_frame(render(CAM, [])), CAM, POSE0, 0.0).reason == "behind_camera"


# ---- the tracker's camera input -------------------------------------------------------------------------------------
def test_the_camera_updates_a_reported_track_and_shrinks_its_cross_uncertainty():
    tr, rgb = tracker_with_box(4)
    tk = tr.tracks[0]
    assert tk.state == TrackState.CONFIRMED
    p_before = tk.filter.P[1, 1]
    res = tr.step_camera(rgb, POSE0, 3 * DT, CAM)
    assert [r.track_id for r in res] == [tk.id] and res[0].reason in ("ok", "same_time")
    assert tk.camera_hits == 1 and tk.filter.P[1, 1] < p_before and abs(res[0].innovation_rad) < 0.03


def test_tentative_tracks_get_no_camera_update_and_a_disabled_camera_does_nothing():
    tr, rgb = tracker_with_box(1)
    assert tr.tracks[0].state == TrackState.TENTATIVE and tr.step_camera(rgb, POSE0, 0.0, CAM) == []
    off, rgb = tracker_with_box(4, Tracker(TrackerConfig(bearing=BearingConfig(enabled=False))))
    assert off.step_camera(rgb, POSE0, 3 * DT, CAM) == [] and off.tracks[0].camera_hits == 0


def test_a_coasting_track_is_updated_by_the_camera_without_resetting_its_lidar_misses():
    tr, rgb = tracker_with_box(4)
    tr.step([], 4 * DT)
    tk = tr.tracks[0]
    assert tk.state == TrackState.COASTING and tk.misses == 1
    res = tr.step_camera(rgb, POSE0, 4 * DT, CAM)
    assert res[0].reason in ("ok", "same_time") and tk.camera_hits == 1 and tk.misses == 1 and tk.state == TrackState.COASTING


def test_a_bearing_that_disagrees_with_the_track_is_gated():
    tr, _ = tracker_with_box(4)
    other = render(CAM, [(25.0, -8.0, 4.5, 1.8, 1.5)])                 # the object in the image is far to the right of the track
    res = tr.step_camera(other, POSE0, 3 * DT, CAM)
    assert res[0].reason in ("gated", "no_edge", "ambiguous", "implausible", "image_border", "low_contrast") and tr.tracks[0].camera_hits == 0


def test_the_lidar_step_after_a_camera_update_with_a_later_time_stamp_does_not_fail():
    tr, rgb = tracker_with_box(4)
    tr.step_camera(rgb, POSE0, 4 * DT + 0.03, CAM)                    # the camera frame is stamped after the next scan
    tr.step([], 3 * DT + 0.01)                                        # a miss at a scan time before the track time: no predict
    res = tr.step([wc(BOX[0], BOX[1], 5 * DT, width=BOX[3])], 5 * DT)
    assert len(res.matches) == 1 and len(tr.tracks) == 1


def test_colour_edges_find_an_object_that_has_the_gray_level_of_its_background():
    img = np.full((120, 160, 3), 128, dtype=np.uint8)
    img[40:80, 60:100] = (200, 116, 0)                                        # gray level 128 as well
    assert prepare_frame(img, BearingConfig(use_color=False)).edges.sum() == 0
    frame = prepare_frame(img, BearingConfig(use_color=True))
    assert frame.edges[40:80, 55:65].any() and frame.edges[40:80, 95:105].any()


def test_a_flat_gray_ROI_is_low_contrast_but_a_colour_only_object_is_not():
    from server.tracking.bearing import _contrast
    img = np.full((120, 160, 3), 128, dtype=np.uint8)
    img[40:80, 60:100] = (200, 116, 0)
    assert _contrast(prepare_frame(img), 30, 90, 50, 110, BearingConfig(use_color=True)) > 6.0
    assert _contrast(prepare_frame(img), 30, 90, 50, 110, BearingConfig(use_color=False)) < 6.0