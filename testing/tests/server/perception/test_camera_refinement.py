"""Tests for the camera edge refinement (step 5.6): synthetic scenes seen by the LiDAR and by a synthetic camera."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from server.communication import protocol as P
from server.perception.camera_refinement import (CameraBuffer, CameraRefiner, RefinementConfig, apply_refinement,
                                                 to_camera_vehicle)
from server.perception.cluster_geometry import GeometryExtractor
from server.perception.ground import GroundRemover
from server.perception.segmentation import Segmenter
from testing.support.synthetic_camera import box_rect, make_camera, render
from testing.support.synthetic_scene import Scene, make_lidar

CAMERA = make_camera()
LIDAR = make_lidar()
POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
CAM_HEADER = SimpleNamespace(pose=POSE, t=0.0)


def lidar_clusters(x, y, length, width, height):
    img, truth, geo = Scene().add_box(x, y, length, width, height).render(LIDAR)
    seg = Segmenter(geo).segment(img, GroundRemover(geo).candidates(img))
    return img, geo, GeometryExtractor(geo).extract_all(img, seg)


def refine(img, geo, geoms, rgb, config=None):
    return CameraRefiner(CAMERA, geo, config).refine(geoms, img, CAM_HEADER, rgb)


def test_the_edges_the_lidar_missed_are_found():
    """The LiDAR sees a 1.4 m wide car, the camera shows the 1.8 m wide one."""
    img, geo, geoms = lidar_clusters(20, 0, 4.5, 1.4, 1.5)
    rgb = render(CAMERA, [(20, 0, 4.5, 1.8, 1.5)])
    (r,) = refine(img, geo, geoms, rgb)
    u0, _, u1, _ = box_rect(CAMERA, 20, 0, 4.5, 1.8, 1.5)
    assert r.used_left and r.used_right
    assert r.u_refined[0] == pytest.approx(u0, abs=2.0) and r.u_refined[1] == pytest.approx(u1, abs=2.0)
    assert r.width_lidar_m < 1.6 and r.width_refined_m == pytest.approx(1.8, abs=0.15)
    assert r.shift_left_m == pytest.approx(0.0, abs=0.1)


def test_an_object_that_is_off_to_one_side_shifts_the_centre_towards_it():
    img, geo, geoms = lidar_clusters(20, 0, 4.5, 1.4, 1.5)
    rgb = render(CAMERA, [(20, 0.3, 4.5, 1.4, 1.5)])                    # the camera shows it 0.3 m further left
    (r,) = refine(img, geo, geoms, rgb)
    assert r.applied and r.shift_left_m == pytest.approx(0.3, abs=0.12)
    g = apply_refinement(geoms[0], r)
    assert g.centre_xy[1] - geoms[0].centre_xy[1] == pytest.approx(r.shift_left_m, abs=0.02)
    assert g.width_m == r.width_refined_m and g.nearest_point is geoms[0].nearest_point      # only the lateral extent changes


def test_a_plain_image_falls_back_to_the_lidar():
    img, geo, geoms = lidar_clusters(20, 0, 4.5, 1.8, 1.5)
    (r,) = refine(img, geo, geoms, render(CAMERA, flat=True, noise=0.5))
    assert not r.applied and r.reason_left == r.reason_right == "low_contrast"
    assert r.width_refined_m == r.width_lidar_m
    assert apply_refinement(geoms[0], r) is geoms[0]


def test_two_comparable_edges_are_ambiguous():
    img, geo, geoms = lidar_clusters(20, 0, 4.5, 1.8, 1.5)
    u0, _, u1, _ = box_rect(CAMERA, 20, 0, 4.5, 1.8, 1.5)
    rgb = render(CAMERA, [(20, 0, 4.5, 1.8, 1.5)], vertical_bars=[(u0 - 7, u0 - 5)])         # a pole just left of the car
    (r,) = refine(img, geo, geoms, rgb)
    assert not r.used_left and r.reason_left == "ambiguous"
    assert r.used_right                                                                       # the other side is still fine


def test_a_side_near_the_image_border_falls_back():
    img, geo, geoms = lidar_clusters(20, 15.5, 4.5, 1.8, 1.5)                  # far to the left, near the edge of the view
    rgb = render(CAMERA, [(20, 15.5, 4.5, 1.8, 1.5)])
    (r,) = refine(img, geo, geoms, rgb)
    assert not r.used_left and r.reason_left == "image_border"


def test_no_camera_frame_and_the_toggle():
    img, geo, geoms = lidar_clusters(20, 0, 4.5, 1.8, 1.5)
    refiner = CameraRefiner(CAMERA, geo)
    (r,) = refiner.refine(geoms, img, None, None)
    assert not r.applied and r.reason_left == "no_camera_frame"
    off = CameraRefiner(CAMERA, geo, RefinementConfig(enabled=False))
    out, refs = off.refine_and_apply(geoms, img, CAM_HEADER, render(CAMERA, [(20, 0, 4.5, 1.8, 1.5)]))
    assert out is geoms and refs == []


def test_the_motion_between_scan_and_camera_frame_is_compensated():
    p = np.array([10.0, 0.0, 1.0])
    assert to_camera_vehicle(p, POSE, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)) == pytest.approx([9.0, 0.0, 1.0])     # car moved 1 m on
    assert to_camera_vehicle(p, POSE, (0.0, 0.0, 0.0, math.pi / 2, 0.0, 0.0)) == pytest.approx([0.0, -10.0, 1.0])
    same = (5.0, 5.0, 0.0, math.pi / 2, 0.0, 0.0)
    assert to_camera_vehicle(p, same, same) == pytest.approx(p)


def test_the_camera_buffer_returns_the_closest_frame_within_the_gap():
    buf = CameraBuffer(max_frames=3)
    rgb = lambda value: np.full((4, 6, 3), value, dtype=np.uint8)
    for seq, t in enumerate([0.0, 0.0333, 0.0667, 0.1]):
        payload = P.camera_payload(6, 4, P.CameraFormat.RAW, rgb(seq * 10).tobytes())
        buf.add(P.build_message(P.MsgType.CAMERA, seq, payload, t=t))
    h, img = buf.nearest(0.07, 0.02)
    assert h.seq == 2 and (img == 20).all()
    assert buf.nearest(0.5, 0.02) is None                               # nothing close enough
    assert buf.nearest(0.0, 0.02) is None                               # frame 0 was pushed out (3 frames kept)