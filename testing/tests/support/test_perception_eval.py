"""Tests for the evaluation metrics (testing/support/perception_eval.py) and the ground-truth reader."""
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from testing.support.ground_truth import GroundTruth, TruthObject
from testing.support.perception_eval import EvalConfig, associate, evaluate_scan, summarise

SENSOR = np.array([1.3, 0.0])


def truth(id=1, x=20.0, y=0.0, length=4.0, width=2.0, yaw=0.0, hits=30, in_view=True, label="car"):
    return TruthObject(id, label, np.array([x, y]), yaw, length, width, 1.5, hits, in_view)


def cluster(label, xs, ys, centre=None, width=1.0, nearest=None):
    pts = np.stack([xs, ys, np.full(len(xs), 1.0)], axis=1)
    near = pts[np.argmin(np.hypot(pts[:, 0] - SENSOR[0], pts[:, 1] - SENSOR[1]))] if nearest is None else nearest
    mean = pts[:, :2].mean(axis=0)
    return SimpleNamespace(label=label, points=pts, nearest_point=near, n_cells=len(xs), width_m=width,
                           centre_xy=mean if centre is None else np.asarray(centre), centroid_xy=mean)


# ---- TruthObject --------------------------------------------------------------------------------------------------------
def test_footprint_contains_and_distance():
    t = truth(x=20.0, y=0.0, length=4.0, width=2.0)
    assert t.contains(np.array([[20.0, 0.0], [21.9, 0.9], [22.5, 0.0], [20.0, 1.2]]), 0.0).tolist() == [True, True, False, False]
    assert t.contains(np.array([[22.3, 0.0]]), 0.4).all()                        # the margin
    assert t.distance_from(SENSOR) == pytest.approx(20.0 - 2.0 - 1.3)               # to the near face
    assert t.distance_from(np.array([20.0, 0.0])) == 0.0                              # inside
    assert t.lateral_extent(np.array([0.0, 0.0])) == pytest.approx(2.0, abs=0.05)    # seen from behind: the width
    turned = truth(x=20.0, yaw=math.pi / 2)
    assert turned.lateral_extent(np.array([0.0, 0.0])) == pytest.approx(4.0, abs=0.2)  # seen from the side: the length
    assert turned.footprint().shape == (4, 2)


# ---- association and the scan metrics ------------------------------------------------------------------------------------
def test_a_cluster_goes_to_the_hazard_that_holds_most_of_its_points():
    a, b = truth(1, x=20.0, y=0.0), truth(2, x=20.0, y=5.0)
    c = cluster(0, np.array([19.0, 20.0, 21.0]), np.array([0.0, 0.2, -0.2]))
    far = cluster(1, np.array([40.0, 41.0]), np.array([10.0, 10.0]))
    assert associate([c, far], [a, b], EvalConfig()) == [0, None]


def test_evaluate_scan_found_missed_fragmented_and_false_cluster():
    car, person, ghost = truth(1, 20.0, 0.0), truth(2, 35.0, 4.0, 0.6, 0.5, label="person"), truth(3, 50.0, -6.0, hits=1)
    clusters = [cluster(0, np.array([18.5, 18.5, 18.5]), np.array([-0.8, 0.0, 0.8]), centre=[19.5, 0.0], width=1.8),   # the car, twice
                cluster(1, np.array([21.0, 21.5, 21.8]), np.array([-0.8, 0.0, 0.8])),
                cluster(2, np.array([60.0, 60.2]), np.array([3.0, 3.0]))]                                             # nothing there
    haz, cl = evaluate_scan(0, 0.0, clusters, [car, person, ghost], SENSOR, EvalConfig(min_hits=2))
    by_id = {r["id"]: r for r in haz}
    assert by_id[1]["detected"] and by_id[1]["n_clusters"] == 2 and by_id[1]["detectable"]
    assert by_id[1]["centre_err"] == pytest.approx(0.5, abs=0.01)                    # best cluster: (19.5, 0) against (20, 0)
    assert by_id[1]["nearest_err"] == pytest.approx(0.5, abs=0.01)                    # nearest point 18.5, the car's face is at 18
    assert not by_id[2]["detected"] and by_id[2]["detectable"]                        # a missed person
    assert not by_id[3]["detectable"]                                                 # 1 cell: not a target
    assert [r["matched"] for r in cl] == [True, True, False]


def _rows(detected, ranges, labels=None, hits=None):
    return pd.DataFrame(dict(scan=range(len(detected)), detectable=True, detected=detected, true_range_m=ranges,
                             label=labels or ["car"] * len(detected), hits=hits or [20] * len(detected), in_view=True,
                             n_clusters=[1 if d else 0 for d in detected], centre_err=0.5, centroid_err=1.0, nearest_err=0.01,
                             width_err=-0.1))


def test_summary_numbers_and_bins():
    haz = _rows([True, True, False, True], [10.0, 30.0, 70.0, 90.0], hits=[40, 12, 2, 1])
    cl = pd.DataFrame(dict(matched=[True, True, False]))
    s = summarise(haz, cl)
    assert s["recall"] == pytest.approx(0.75) and s["precision"] == pytest.approx(2 / 3)
    assert s["unmatched_clusters"] == 1 and s["fragmented_fraction"] == 0.0
    assert s["recall_by_range"]["[0.0, 20.0)"] == (1.0, 1) and s["recall_by_range"]["[60.0, 80.0)"] == (0.0, 1)
    assert s["recall_by_hits"]["[2.0, 3.0)"] == (0.0, 1)
    assert s["centre_err"]["median"] == 0.5 and s["nearest_err"]["max_abs"] == pytest.approx(0.01)


# ---- ground truth reader ---------------------------------------------------------------------------------------------------
def frames(ego_x, ego_y, ego_yaw, hx, hy):
    t = [0.0, 1.0]
    ego = pd.DataFrame(dict(t=t, x=[ego_x] * 2, y=[ego_y] * 2, z=0, yaw_rad=[ego_yaw] * 2))
    haz = pd.DataFrame(dict(t=t, id=1, x=[hx] * 2, y=[hy] * 2, z=0.75, yaw_rad=ego_yaw, vx=0, vy=0, speed=0,
                            length=4.0, width=2.0, height=1.5))
    info = pd.DataFrame(dict(id=[1], name=["car"], label=["car"], length=4.0, width=2.0, height=1.5, colliders=1))
    vis = pd.DataFrame(dict(t=[0.5], scan=0, id=1, hits_geometric=30, hits_returned=28, min_range_m=5.0, centre_range_m=10.0,
                            azimuth_deg=0.0, in_view=True))
    return GroundTruth(ego, haz, info, vis)


def test_truth_is_moved_into_the_vehicle_frame_of_the_ego():
    # the ego at (10, 5) faces +y; the hazard is 10 m ahead of it, at (10, 15), and 3 m to its right (x = 13)
    gt = frames(10.0, 5.0, math.pi / 2, 13.0, 15.0)
    (obj,) = gt.objects_at(0.5)
    assert obj.centre == pytest.approx([10.0, -3.0])
    assert obj.yaw == pytest.approx(0.0)                                          # it faces the same way as the ego
    assert obj.hits_returned == 28 and obj.in_view and obj.label == "car"
    assert gt.objects_at(0.8)[0].hits_returned is None                           # no visibility row near that time
    assert gt.objects_at(7.0) == []                                              # outside the recorded time


def test_a_person_swallowed_by_the_neighbouring_car_cluster_is_reported_as_merged():
    car, person = truth(1, 20.0, 0.0), truth(2, 20.0, 2.0, 0.6, 0.5, label="person")
    xs = np.array([18.5, 19.0, 19.5, 20.0, 20.3, 20.3, 20.3])
    ys = np.array([-0.5, 0.0, 0.5, 0.0, 1.8, 2.0, 2.2])                      # last three points are on the person
    haz, _ = evaluate_scan(0, 0.0, [cluster(0, xs, ys)], [car, person], SENSOR, EvalConfig(min_hits=2))
    by_id = {r["id"]: r for r in haz}
    assert by_id[1]["detected"] and not by_id[1]["merged"]
    assert not by_id[2]["detected"] and by_id[2]["merged"]
    assert by_id[2]["stolen_pts"] >= 2