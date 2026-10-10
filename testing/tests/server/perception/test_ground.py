"""Tests for ground removal (step 5.3), on synthetic scenes ray-cast by testing/support/synthetic_scene.py."""
import math

import numpy as np
import pytest

from server.perception.ground import GroundConfig, GroundMethod, GroundRemover, _isolated_runs
from server.perception.segmentation import beta_labels
from testing.support.synthetic_scene import Scene, make_lidar
from testing.support.paths import RECORDINGS_DIR

METHODS = [GroundMethod.HEIGHT, GroundMethod.RANGE_IMAGE]
LIDAR = make_lidar()


def run(scene, method, **render):
    img, truth, geo = scene.render(LIDAR, **render)
    return img, truth, geo, GroundRemover(geo, GroundConfig(method=method)).ground_mask(img)


def ground_recall(truth, mask):
    gt = truth == 1
    return (mask & gt).sum() / gt.sum()


def slope(deg):
    return math.tan(math.radians(deg))


@pytest.mark.parametrize("method", METHODS)
def test_flat_ground_is_removed(method):
    img, truth, geo, mask = run(Scene(), method)
    assert (truth == 1).sum() > 1000                       # the scene really has a lot of ground in it
    assert ground_recall(truth, mask) > 0.99
    assert not (mask & (truth != 1)).any()                 # sky and no-return cells are never ground


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("scene", [Scene().add_box(20, 0, 4.5, 1.8, 1.5),         # a car
                                   Scene().add_box(12, 0, 0.6, 0.5, 1.7),          # a person
                                   Scene().add_box(30, 0, 0.6, 60.0, 3.0)],        # a wall across the road
                         ids=["car", "person", "wall"])
def test_things_standing_on_the_road_are_kept(method, scene):
    img, truth, geo, mask = run(scene, method)
    z = geo.points_vehicle(img)[..., 2]
    obj = truth == 2
    assert not (mask & obj & (z > 0.3)).any()              # nothing higher than 30 cm is ever called ground
    assert (obj & ~mask).sum() / obj.sum() > 0.8           # at most the lowest rim of the object is lost
    assert ground_recall(truth, mask) > 0.99


def test_a_constant_slope_is_followed_by_the_range_image_method_only():
    a = 4.0
    scene = Scene([(0.0, slope(a))])
    _, truth, _, ri = run(scene, GroundMethod.RANGE_IMAGE, pitch=math.radians(a))
    _, _, _, hf = run(scene, GroundMethod.HEIGHT, pitch=math.radians(a))
    assert ground_recall(truth, ri) > 0.99
    assert ground_recall(truth, hf) < 0.1                  # the flat plane is left behind: the road is reported as obstacles


def test_a_hill_ahead_of_a_flat_road():
    scene = Scene([(0.0, 0.0), (20.0, slope(8.0))])
    _, truth, _, ri = run(scene, GroundMethod.RANGE_IMAGE)
    _, _, _, hf = run(scene, GroundMethod.HEIGHT)
    assert ground_recall(truth, ri) > 0.99
    assert ground_recall(truth, hf) < 0.7                  # the hill itself is reported as an obstacle


def test_a_crest():
    scene = Scene([(0.0, slope(5.0)), (30.0, -slope(5.0))])
    _, truth, _, ri = run(scene, GroundMethod.RANGE_IMAGE, pitch=math.radians(5.0))
    _, _, _, hf = run(scene, GroundMethod.HEIGHT, pitch=math.radians(5.0))
    assert ground_recall(truth, ri) > 0.99
    assert ground_recall(truth, hf) < 0.5                  # scenario 9: the height filter fails here


def test_a_slope_steeper_than_the_tolerance_is_not_ground():
    scene = Scene([(0.0, 0.0), (20.0, slope(20.0))])
    img, truth, geo, mask = run(scene, GroundMethod.RANGE_IMAGE)
    ramp = (truth == 1) & (geo.points_vehicle(img)[..., 0] > 22.0)
    assert ramp.sum() > 100 and not (mask & ramp).any()


def test_the_height_filter_uses_the_pitch_of_the_car():
    scene = Scene()                                        # flat road, the nose is pitched 3 degrees down
    pitch = -math.radians(3.0)
    _, truth, _, right = run(scene, GroundMethod.HEIGHT, pitch=pitch)
    _, _, _, wrong = run(scene, GroundMethod.HEIGHT, pitch=pitch, pose_pitch=0.0)
    assert ground_recall(truth, right) > 0.99
    assert ground_recall(truth, wrong) < 0.5               # without the pitch correction the road ahead looks like a hill


@pytest.mark.parametrize("method", METHODS)
def test_dropouts_do_not_break_the_ground(method):
    img, truth, geo, mask = run(Scene().add_box(25, 0, 4.5, 1.8, 1.5), method, dropout=0.3)
    assert ground_recall(truth, mask) > 0.99
    assert not (mask & ~img.valid).any()                   # only cells with a return can be ground


def test_candidates_are_the_valid_cells_that_are_not_ground():
    img, _, geo = Scene().add_box(20, 0, 4.5, 1.8, 1.5).render(LIDAR, dropout=0.1)
    remover = GroundRemover(geo)
    assert remover.method is GroundMethod.RANGE_IMAGE      # the default
    cand, ground = remover.candidates(img), remover.ground_mask(img)
    assert np.array_equal(cand, img.valid & ~ground)
    assert cand.any() and not (cand & ground).any()


def test_there_is_no_way_to_switch_ground_removal_off():
    _, _, geo = Scene().render(LIDAR)
    for bad in ("none", "off", None):
        with pytest.raises(ValueError):
            GroundRemover(geo, GroundConfig(method=bad))
    assert GroundRemover(geo, GroundConfig(method="height")).method is GroundMethod.HEIGHT


def _clusters(img, geo, truth, use):
    labels = beta_labels(img.range_m, geo.alpha_h, geo.alpha_v, use)
    ids, sizes = np.unique(labels[labels >= 0], return_counts=True)
    return labels, ids, sizes


def test_unremoved_ground_rows_form_clusters():
    """The reason ground removal is needed on every road: on a flat, empty road the near-horizon beams return ground
    rows out to the range limit and the beta criterion turns them into big objects."""
    img, truth, geo = Scene().render(LIDAR)
    _, ids, sizes = _clusters(img, geo, truth, img.valid)                       # no ground removal
    assert len(ids) >= 5 and sizes.max() > 500
    for method in METHODS:
        mask = GroundRemover(geo, GroundConfig(method=method)).candidates(img)
        assert len(_clusters(img, geo, truth, mask)[1]) == 0                    # nothing is left to cluster


def test_unremoved_ground_merges_with_a_person_on_the_road():
    img, truth, geo = Scene().add_box(30, 0, 0.6, 0.5, 1.7).render(LIDAR)
    person = truth == 2
    labels, ids, sizes = _clusters(img, geo, truth, img.valid)
    merged = np.unique(labels[person])
    assert len(merged) == 1                                                     # one cluster, and ...
    assert ((labels == merged[0]) & (truth == 1)).sum() > 100                   # ... most of it is road
    for method in METHODS:
        use = GroundRemover(geo, GroundConfig(method=method)).candidates(img)
        labels, ids, sizes = _clusters(img, geo, truth, use)
        assert len(ids) == 1 and not ((labels >= 0) & (truth == 1)).any()      # only the person is left


@pytest.mark.skipif(not (RECORDINGS_DIR / "dev_default_seed12345" / "lidar.bin").exists(),
                    reason="no development recording in testing/data/recordings")
@pytest.mark.parametrize("method", METHODS)
def test_a_recorded_scan_runs_through_both_methods(method):
    from server.communication import protocol as P
    from server.geometry.calibration import Calibration
    from server.perception.range_image import RangeImageBuilder
    from testing.support.dev_recording import load_lidar
    d = RECORDINGS_DIR / "dev_default_seed12345"
    calib = Calibration.from_json(d / "calibration.json")
    rec = load_lidar(d)[0]
    b = RangeImageBuilder(calib.lidar)
    msg = P.build_message(P.MsgType.LIDAR, int(rec["frame"]), P.lidar_payload(rec["ranges"]), t=float(rec["t"]),
                          pose=tuple(rec["pose"]), speed=0.0, yaw_rate=0.0, steering_angle=0.0)
    img = b.ingest(msg)
    mask = GroundRemover(b.geometry, GroundConfig(method=method)).ground_mask(img)
    assert mask.shape == img.shape and mask.dtype == bool
    assert not (mask & ~img.valid).any()
    assert mask.any()                                                           # there is road in a driving scene


def test_a_person_far_away_is_not_swallowed_by_the_ground():
    """78 m away the beams are 10 m apart on the ground, so the lowest cell of a person is only a gentle rise from the last
    ground cell. The person has two cells there; both must stay candidates (found by the evaluation of step 5.7)."""
    img, truth, geo, mask = run(Scene().add_box(78, 0, 0.6, 0.5, 1.7), GroundMethod.RANGE_IMAGE)
    person = truth == 2
    assert person.sum() == 2 and not (mask & person).any()
    assert ground_recall(truth, mask) > 0.99


def test_runs_of_returns_with_max_range_on_both_sides_are_isolated():
    valid = np.array([0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0], dtype=bool)
    maxr = ~valid
    assert list(np.flatnonzero(_isolated_runs(valid, maxr, 4))) == [1, 2, 12]       # the run of 6 is too long to be an object
    assert not _isolated_runs(valid, np.zeros_like(valid), 4).any()                # no max range next to them: not isolated
    assert not _isolated_runs(np.array([1, 1, 0], dtype=bool), np.array([0, 0, 1], dtype=bool), 4).any()   # at the image edge