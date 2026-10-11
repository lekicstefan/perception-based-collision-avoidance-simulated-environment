"""Existence (hit ratio), the longer life of reliable tracks and the camera keeping a coasting track (step 6.7)."""
import pytest

from server.tracking.track import TrackConfig
from server.tracking.tracker import Tracker, TrackerConfig
from testing.tests.server.tracking.test_tracker import DT, run


def misses_until_deleted(tracker, start_k, limit=40, camera=False):
    for j in range(1, limit + 1):
        t = (start_k + j) * DT
        res = tracker.step([], t)
        if res.deleted:
            return j
        if camera:
            tracker.tracks[0].camera_hit(t)
    return None


def test_the_hit_ratio_is_taken_over_the_window():
    tr = Tracker(TrackerConfig(track=TrackConfig(existence_window=10)))
    run(tr, [[(20.0, 0.0)]] * 6)
    tr.step([], 6 * DT)
    tr.step([], 7 * DT)
    t = tr.tracks[0]
    assert t.hit_ratio == pytest.approx(6 / 8) and t.reliability == pytest.approx(1.0)       # before the run of misses: always seen
    for k in range(8, 14):
        tr.step([], k * DT)
        if not tr.tracks:
            break
    assert tr.tracks == [] or tr.tracks[0].hit_ratio <= 0.6


def test_a_young_track_is_deleted_after_five_misses():
    tr = Tracker()
    run(tr, [[(20.0, 0.0)]] * 6)
    assert misses_until_deleted(tr, 5) == 5


def test_an_old_reliable_track_survives_five_more_misses():
    tr = Tracker()
    run(tr, [[(20.0, 0.0)]] * 30)
    assert misses_until_deleted(tr, 29) == 10


def test_an_old_track_that_was_often_missed_gets_no_bonus():
    tr = Tracker()
    run(tr, [[(20.0, 0.0)], [], [(20.0, 0.0)], [(20.0, 0.0)], [], [(20.0, 0.0)], [(20.0, 0.0)], []] * 4 + [[(20.0, 0.0)]] * 4)
    t = tr.tracks[0]
    assert t.age_scans >= 20 and t.reliability < 0.8
    assert misses_until_deleted(tr, 35) == 5


def test_a_recent_camera_bearing_keeps_a_coasting_track_alive_up_to_three_times_k():
    tr = Tracker()
    run(tr, [[(20.0, 0.0)]] * 6)
    tr.tracks[0].camera_hit(5 * DT)
    assert misses_until_deleted(tr, 5, camera=True) == 15
    tr2 = Tracker()
    run(tr2, [[(20.0, 0.0)]] * 6)
    assert misses_until_deleted(tr2, 5) == 5                                                  # without the camera: K