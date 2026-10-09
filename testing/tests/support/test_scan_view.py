"""The scan pictures: colours, shading and that a whole picture can be made (testing/support/scan_view.py)."""
import numpy as np

from server.geometry.calibration import Calibration, EgoCalibration
from server.perception.pipeline import PerceptionPipeline
from testing.support import scan_view
from testing.support.synthetic_camera import make_camera, render
from testing.support.synthetic_scene import Scene, make_lidar

LIDAR, CAMERA = make_lidar(), make_camera()
CALIB = Calibration("t", EgoCalibration(2.7, 4.4, 1.8, 1.5, 0.9, 0.8), CAMERA, LIDAR, 100.0)


def result_for(scene, boxes=()):
    pipe = PerceptionPipeline(CALIB)
    img, _, geo = scene.render(LIDAR)
    rgb = render(CAMERA, boxes)
    return pipe.process(img, type("H", (), {"t": img.t, "pose": img.pose})(), rgb), geo


def test_closer_is_lighter_and_farther_is_darker():
    near, far = scan_view.shade(0.3, [1.0, 0.0])
    assert near.astype(int).sum() > far.astype(int).sum()
    assert near.max() == 255 and far.max() < 100                       # the value channel goes from 1.0 to 0.3


def test_every_cluster_has_its_own_colour_and_shading_follows_the_range():
    scene = Scene().add_box(20, 0, 4.5, 1.8, 1.5).add_box(30, 5, 0.6, 0.5, 1.7)
    result, _ = result_for(scene)
    assert len(result.segmentation.clusters) == 2
    col = scan_view.cell_colours(result)
    a, b = result.segmentation.clusters
    hues = {c.label: np.unique(col[c.rows, c.cols].reshape(-1, 3), axis=0) for c in (a, b)}
    assert not set(map(tuple, hues[a.label])) & set(map(tuple, hues[b.label]))
    rng = result.image.range_m[a.rows, a.cols]
    bright = col[a.rows, a.cols].astype(int).sum(axis=1)
    assert bright[np.argmin(rng)] > bright[np.argmax(rng)]               # the near end of the car is lighter
    assert (col[result.ground] != 0).any() and (col[~result.image.valid & ~result.ground] <= 48).all()


def test_a_whole_picture_is_made(tmp_path):
    result, geo = result_for(Scene().add_box(20, 0, 4.5, 1.8, 1.5), boxes=[(20, 0, 4.5, 1.8, 1.5)])
    rgb = render(CAMERA, [(20, 0, 4.5, 1.8, 1.5)])
    panels = [scan_view.range_panel(result, CALIB), scan_view.top_view(result, CALIB, geo), scan_view.camera_panel(result, rgb)]
    assert all(p.dtype == np.uint8 and p.shape[2] == 3 for p in panels)
    picture = scan_view.compose(panels)
    assert picture.shape[0] == panels[0].shape[0] + max(p.shape[0] for p in panels[1:])      # range image, then a row
    path = tmp_path / "scan.png"
    scan_view.save_rgb(path, picture)
    assert path.stat().st_size > 5000