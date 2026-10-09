"""The whole perception pipeline on one synthetic scan (server/perception/pipeline.py)."""
from types import SimpleNamespace

import numpy as np

from server.communication import protocol as P
from server.geometry.calibration import Calibration, EgoCalibration
from server.perception.camera_refinement import RefinementConfig
from server.perception.ground import GroundConfig, GroundMethod
from server.perception.pipeline import PerceptionConfig, PerceptionPipeline
from testing.support.synthetic_camera import make_camera, render
from testing.support.synthetic_scene import Scene, make_lidar

LIDAR, CAMERA = make_lidar(), make_camera()
CALIB = Calibration("t", EgoCalibration(2.7, 4.4, 1.8, 1.5, 0.9, 0.8), CAMERA, LIDAR, 100.0)
BOXES = [(20, 0, 4.5, 1.8, 1.5), (30, 5, 0.6, 0.5, 1.7)]


def scan():
    scene = Scene()
    for b in BOXES:
        scene.add_box(*b)
    return scene.render(LIDAR)[0]


def test_both_objects_are_found_with_and_without_the_camera_and_with_each_ground_method():
    for method in GroundMethod:
        for refine in (False, True):
            cfg = PerceptionConfig(ground=GroundConfig(method=method), refinement=RefinementConfig(enabled=refine))
            pipe = PerceptionPipeline(CALIB, cfg)
            img = scan()
            hdr = SimpleNamespace(t=img.t, pose=img.pose) if refine else None
            result = pipe.process(img, hdr, render(CAMERA, BOXES) if refine else None)
            assert len(result.clusters) == 2
            assert bool(result.refinements) == refine
            assert sorted(round(g.nearest_point[0]) for g in result.clusters) == [18, 30]


def test_a_lidar_message_takes_the_camera_frame_from_the_buffer():
    pipe = PerceptionPipeline(CALIB)
    img = scan()
    rgb = render(CAMERA, BOXES)
    pipe.camera_buffer.add(P.build_message(P.MsgType.CAMERA, 1, P.camera_payload(640, 360, P.CameraFormat.RAW, rgb.tobytes()),
                                           t=img.t + 0.01))
    ranges = np.where(img.valid, np.nan_to_num(img.range_m) * 100, 0).round().astype("uint16")
    raw = P.build_message(P.MsgType.LIDAR, 1, P.lidar_payload(ranges), t=img.t)
    result = pipe.process_message(raw)
    assert result.refinements                                                   # the buffered frame was used