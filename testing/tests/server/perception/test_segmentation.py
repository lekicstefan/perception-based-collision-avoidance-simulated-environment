"""Tests for the angle-based segmentation (step 5.4), on synthetic scenes (testing/support/synthetic_scene.py)."""
import math

import numpy as np
import pytest

from server.perception.ground import GroundRemover
from server.perception.segmentation import Segmenter, SegmentationConfig, beta_angle, beta_labels
from testing.support.paths import RECORDINGS_DIR
from testing.support.synthetic_scene import Scene, make_lidar

LIDAR = make_lidar()
CAR_FRONT_X = 17.75                                         # a car box of length 4.5 centred at x = 20


def segment(scene, config=None):
    img, truth, geo = scene.render(LIDAR)
    candidates = GroundRemover(geo).candidates(img)
    return img, truth, geo, candidates, Segmenter(geo, config).segment(img, candidates)


def overlap(seg, truth, cluster):
    """(cells of the cluster that are really object, object cells in the scene)"""
    return int((truth[cluster.rows, cluster.cols] == 2).sum()), int((truth == 2).sum())


# ---- beta criterion ---------------------------------------------------------------------------------------------------
def test_beta_of_a_face_on_surface_is_large_and_of_a_depth_step_is_small():
    alpha = math.radians(0.2)
    assert math.degrees(beta_angle(10.0, 10.0, alpha)) == pytest.approx(90.0 - 0.1, abs=0.01)    # same range
    assert math.degrees(beta_angle(10.0, 20.0, alpha)) == pytest.approx(0.2, abs=0.01)           # 10 m step
    assert beta_angle(10.0, 20.0, alpha) == pytest.approx(beta_angle(20.0, 10.0, alpha))         # symmetric


def test_beta_labels_on_a_tiny_image():
    r = np.array([[10.0, 10.0, 10.0, 30.0, 30.0],
                  [10.0, 10.0, 10.0, 30.0, 30.0]])
    ah, av = np.full((2, 4), math.radians(0.5)), np.full((1, 5), math.radians(0.5))
    labels = beta_labels(r, ah, av, np.ones(r.shape, dtype=bool))
    assert (labels[:, :3] == labels[0, 0]).all() and (labels[:, 3:] == labels[0, 3]).all()
    assert labels[0, 0] != labels[0, 3]                                          # 20 m depth step: two objects
    use = np.ones(r.shape, dtype=bool)
    use[:, 1] = False                                                            # a cell that is not a candidate
    labels = beta_labels(r, ah, av, use)
    assert (labels[:, 1] == -1).all() and labels[0, 0] != labels[0, 2]           # is a boundary


# ---- scenes -----------------------------------------------------------------------------------------------------------
def test_an_empty_road_has_no_clusters():
    assert segment(Scene())[-1].clusters == []


def test_a_car_is_one_cluster():
    img, truth, geo, cand, seg = segment(Scene().add_box(20, 0, 4.5, 1.8, 1.5))
    assert len(seg.clusters) == 1
    inside, total = overlap(seg, truth, seg.clusters[0])
    assert inside == seg.clusters[0].n_cells                  # nothing but car in the cluster
    assert inside / total > 0.85                              # and nearly all of the car
    assert seg.clusters[0].range_m == pytest.approx(16.5, abs=1.0)
    assert (seg.labels[seg.clusters[0].rows, seg.clusters[0].cols] == 0).all()


@pytest.mark.parametrize("x", [12, 30, 60])
def test_a_person_is_one_cluster_at_any_range(x):
    img, truth, geo, cand, seg = segment(Scene().add_box(x, 0, 0.6, 0.5, 1.7))
    assert len(seg.clusters) == 1
    inside, total = overlap(seg, truth, seg.clusters[0])
    assert inside == seg.clusters[0].n_cells and inside >= 2


def test_two_separate_objects_are_two_clusters():
    _, truth, _, _, seg = segment(Scene().add_box(20, 0, 4.5, 1.8, 1.5).add_box(30, 3, 0.6, 0.5, 1.7))
    assert len(seg.clusters) == 2
    near, far = sorted(c.range_m for c in seg.clusters)
    assert 15 < near < 18 and 27 < far < 30


def test_cells_that_are_not_candidates_are_boundaries():
    img, truth, geo = Scene().add_box(30, 0, 0.6, 60.0, 3.0).render(LIDAR)       # a wall across the whole view
    cand = GroundRemover(geo).candidates(img)
    assert len(Segmenter(geo).segment(img, cand).clusters) == 1
    cand[:, 120] = False                                                         # one column without returns
    assert len(Segmenter(geo).segment(img, cand).clusters) == 2


def test_a_person_touching_a_car_is_split_off():
    person_front = CAR_FRONT_X - 0.3                            # 0.3 m in front of the car's front, touching its side
    scene = Scene().add_box(20, 0, 4.5, 1.8, 1.5).add_box(person_front + 0.3, 1.15, 0.6, 0.5, 1.7)
    _, truth, _, _, with_split = segment(scene)
    _, _, _, _, without = segment(scene, SegmentationConfig(split=False))
    assert len(without.clusters) == 1                         # the beta criterion alone merges them
    assert len(with_split.clusters) == 2                      # the depth step separates them
    sizes = sorted(c.n_cells for c in with_split.clusters)
    assert sizes[0] >= 20 and sizes[1] > 100                  # the person, and the whole car


def test_a_wide_flat_wall_is_not_cut_up():
    assert len(segment(Scene().add_box(30, 0, 0.6, 60.0, 3.0))[-1].clusters) == 1


def test_the_minimum_size_depends_on_the_range():
    thin = segment(Scene().add_box(8, 0, 0.12, 0.12, 1.5))                       # a 12 cm post at 8 m: 1 to 2 columns
    assert (thin[1] == 2).any()                                                  # it is hit ...
    assert thin[-1].clusters == [] and thin[-1].n_discarded >= 1                 # ... and is too thin for that range
    far = segment(Scene().add_box(60, 0, 0.6, 0.5, 1.7))                         # a person at 60 m is also 1 to 2 columns
    assert len(far[-1].clusters) == 1                                            # but that is enough at 60 m


def test_the_step_test_ignores_steady_changes():
    step = Segmenter._is_step
    adj = np.ones(5, dtype=bool)
    assert not step(np.array([0.3, 0.3, 0.3, 0.3, 0.3]), adj, 0.15).any()          # a slanted wall
    assert not step(np.array([0.0, 0.0, 0.7, 0.7, 0.7]), adj, 0.15).any()          # the corner of a car
    assert list(np.flatnonzero(step(np.array([0.0, 0.0, 0.4, 0.0, 0.0]), adj, 0.15))) == [2]      # a real step
    assert not step(np.array([0.0, 0.0, 0.1, 0.0, 0.0]), adj, 0.15).any()          # too small


@pytest.mark.skipif(not (RECORDINGS_DIR / "dev_default_seed12345" / "lidar.bin").exists(),
                    reason="no development recording in testing/data/recordings")
def test_a_recorded_scan():
    from server.communication import protocol as P
    from server.geometry.calibration import Calibration
    from server.perception.range_image import RangeImageBuilder
    from testing.support.dev_recording import load_lidar
    d = RECORDINGS_DIR / "dev_default_seed12345"
    calib = Calibration.from_json(d / "calibration.json")
    b = RangeImageBuilder(calib.lidar)
    for rec in load_lidar(d)[:5]:
        msg = P.build_message(P.MsgType.LIDAR, int(rec["frame"]), P.lidar_payload(rec["ranges"]), t=float(rec["t"]),
                              pose=tuple(rec["pose"]), speed=0.0, yaw_rate=0.0, steering_angle=0.0)
        img = b.ingest(msg)
        cand = GroundRemover(b.geometry).candidates(img)
        seg = Segmenter(b.geometry).segment(img, cand)
        assert seg.labels.shape == img.shape
        assert not ((seg.labels >= 0) & ~cand).any()                             # clusters only hold candidate cells
        assert sum(c.n_cells for c in seg.clusters) == int((seg.labels >= 0).sum())