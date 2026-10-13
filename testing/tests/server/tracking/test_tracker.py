"""Track lifecycle, coasting, merge and births of the tracker (step 6.5), on hand-made world clusters."""
import math

import numpy as np
import pytest

from server.tracking.kalman import FilterConfig
from server.tracking.measurement import WorldCluster
from server.tracking.track import TrackConfig, TrackState
from server.tracking.tracker import Tracker, TrackerConfig

DT = 0.1


def wc(x, y, t, width=1.0, depth=1.0):
    """A cluster whose corrected centre is (x, y), seen from a sensor at the origin (its visible box is width x depth around it)."""
    c = np.array([x, y], dtype=float)
    return WorldCluster(t=t, pose=(0.0,) * 6, label=0, n_cells=50, nearest_xy=c - [0.5, 0.0], centre_xy=c, centroid_xy=c,
                        sensor_xy=np.zeros(2), bearing=math.atan2(y, x), range_m=math.hypot(x - 0.5, y), corners_xy=c + np.array([[-depth / 2, -width / 2], [depth / 2, -width / 2], [depth / 2, width / 2], [-depth / 2, width / 2]]),
                        width_m=width, depth_m=depth, height_m=1.5, top_m=1.5)


def run(tracker, scans):
    """scans: list of lists of (x, y); returns the StepResults. Scan k is at t = k * DT."""
    return [tracker.step([wc(x, y, k * DT) for x, y in pts], k * DT) for k, pts in enumerate(scans)]


def test_a_new_track_is_tentative_with_a_large_velocity_uncertainty():
    tr = Tracker()
    res = run(tr, [[(20.0, 0.0)]])[0]
    t = tr.tracks[0]
    assert res.births == [(t.id, 0)] and t.state == TrackState.TENTATIVE and not t.reported
    assert t.filter.P[2, 2] == pytest.approx(FilterConfig().initial_velocity_sigma ** 2)


def test_a_track_is_confirmed_after_three_consecutive_hits():
    tr = Tracker()
    res = run(tr, [[(20.0, 0.0)]] * 4)
    assert [r.confirmed for r in res] == [[], [], [1], []]
    assert tr.tracks[0].state == TrackState.CONFIRMED and tr.tracks[0].reported and tr.tracks[0].hits == 4


def test_a_single_scan_of_noise_never_becomes_a_reported_track():
    tr = Tracker()
    res = run(tr, [[(20.0, 0.0)], [], [], [], []])
    assert tr.tracks == [] and res[3].deleted == [1]                  # tentative: deleted after more than 2 misses in a row
    assert all(not r.confirmed for r in res)


def test_a_miss_between_hits_restarts_the_count_of_consecutive_hits():
    tr = Tracker()
    run(tr, [[(20.0, 0.0)], [(20.0, 0.0)], [], [(20.0, 0.0)], [(20.0, 0.0)]])
    assert tr.tracks[0].state == TrackState.TENTATIVE                  # hits 1, 2, miss, 1, 2: not three in a row
    run_more = tr.step([wc(20.0, 0.0, 0.5)], 0.5)
    assert run_more.confirmed == [1]


def test_a_confirmed_track_coasts_on_its_velocity_and_is_deleted_after_five_misses():
    tr = Tracker(TrackerConfig(track=TrackConfig(coast_bonus_misses=0)))                # the plain rule: K misses
    run(tr, [[(20.0 + 5.0 * DT * k, 0.0)] for k in range(30)])        # 5 m/s along x
    t = tr.tracks[0]
    assert t.filter.speed == pytest.approx(5.0, abs=0.7)
    x_last, t_last = t.position[0], t.filter.t
    for k in range(1, 5):
        res = tr.step([], t_last + k * DT)
        assert tr.tracks[0].state == TrackState.COASTING and tr.tracks[0].reported and not res.deleted
    assert tr.tracks[0].position[0] == pytest.approx(x_last + 5.0 * 4 * DT, abs=0.8)       # it kept moving
    res = tr.step([], t_last + 5 * DT)
    assert res.deleted == [1] and tr.tracks == []


def test_a_coasting_track_that_is_seen_again_keeps_its_identity():
    tr = Tracker()
    run(tr, [[(20.0, 0.0)]] * 6)
    t0 = tr.tracks[0].filter.t
    tr.step([], t0 + DT)
    tr.step([], t0 + 2 * DT)
    res = tr.step([wc(20.0, 0.0, t0 + 3 * DT)], t0 + 3 * DT)
    assert [m[0] for m in res.matches] == [1] and res.births == [] and tr.tracks[0].state == TrackState.CONFIRMED


def test_two_objects_keep_two_tracks_and_their_own_measurements():
    tr = Tracker()
    run(tr, [[(20.0, -3.0), (30.0, 4.0)]] * 6)
    assert len(tr.tracks) == 2 and {t.id for t in tr.reported} == {1, 2}
    res = tr.step([wc(30.0, 4.0, 0.6), wc(20.0, -3.0, 0.6)], 0.6)             # the order of the clusters changes
    assert sorted((tid, j) for tid, j, _ in res.matches) == [(1, 1), (2, 0)]


def test_the_other_part_of_a_split_cluster_starts_a_new_tentative_track():
    tr = Tracker()
    run(tr, [[(20.0, 0.0)]] * 6)
    res = tr.step([wc(20.0, 0.0, 0.6), wc(20.4, 2.5, 0.6)], 0.6)
    assert len(res.matches) == 1 and res.matches[0][0] == 1 and len(res.births) == 1
    assert tr.tracks[1].state == TrackState.TENTATIVE and tr.tracks[0].state == TrackState.CONFIRMED


def test_a_track_whose_cluster_fuses_with_another_is_merged_into_the_older_one():
    tr = Tracker(TrackerConfig(track=TrackConfig(merge_scans=3)))
    run(tr, [[(20.0, 0.0), (21.2, 0.0)]] * 8)
    assert len(tr.tracks) == 2
    hits_before = {t.id: t.hits for t in tr.tracks}
    merged = []
    for k in range(8, 11):                                                    # one cluster between them
        merged += tr.step([wc(20.6, 0.0, k * DT)], k * DT).merged
    assert len(merged) == 1 and len(tr.tracks) == 1
    absorbed, kept = merged[0]
    assert hits_before[kept] >= hits_before[absorbed] and tr.tracks[0].id == kept


def test_separate_objects_far_apart_are_never_merged():
    tr = Tracker()
    res = run(tr, [[(20.0, 0.0), (20.0, 6.0)]] * 5 + [[(20.0, 0.0)]] * 6)
    assert all(not r.merged for r in res)


def test_the_track_remembers_its_largest_extent():
    tr = Tracker()
    tr.step([wc(20.0, 0.0, 0.0, width=4.0, depth=1.5)], 0.0)
    tr.step([wc(20.0, 0.0, DT, width=1.0, depth=0.3)], DT)                    # a partial view
    assert tr.tracks[0].max_width_m == 4.0 and tr.tracks[0].max_depth_m == 1.5