"""The camera bearing pictures (step 6.6)."""
import numpy as np
import pytest

from server.tracking.tracker import Tracker
from testing.support.camera_roi_view import draw_rois
from testing.support.synthetic_recording import write_recording
from testing.tests.server.tracking.test_bearing import BOX, CAM, DT, POSE0, tracker_with_box
from testing.tools.tracking.camera_roi_view import main


def test_the_picture_has_the_frame_and_the_edge_map_with_the_regions_drawn():
    tr, rgb = tracker_with_box(4)
    diag = []
    results = tr.step_camera(rgb, POSE0, 3 * DT, CAM, diag)
    assert len(diag) == 1 and diag[0].roi is not None and diag[0].expected_u is not None and diag[0].u is not None
    img = draw_rois(rgb, diag, results, t=0.3)
    assert img.shape == (rgb.shape[0], 2 * rgb.shape[1], 3) and img.dtype == np.uint8
    assert not np.array_equal(img[:, :rgb.shape[1]], rgb)                      # something was drawn on the frame


def test_the_diagnostics_are_filled_also_when_there_is_no_bearing():
    tr, _ = tracker_with_box(4)
    from testing.support.synthetic_camera import render
    diag = []
    tr.step_camera(render(CAM, []), POSE0, 3 * DT, CAM, diag)
    assert diag[0].theta is None and diag[0].expected_u is not None and diag[0].reason != "ok"


def test_the_tool_writes_pictures_for_the_window(tmp_path_factory, tmp_path):
    folder = write_recording(tmp_path_factory.mktemp("rec") / "synthetic")
    written = main(["--recording", str(folder), "--out", str(tmp_path), "--t-from", "1.0", "--t-to", "1.5", "--camera-every", "3"])
    assert len(written) >= 1 and all(p.exists() for p in written)