"""Derived outputs of the tracks: velocity statistics, ellipse, the static flag and its hysteresis (step 6.7)."""
import math

import numpy as np
import pytest

from server.tracking.outputs import OutputConfig, velocity_stats, zero_velocity_d2
from server.tracking.tracker import Tracker
from testing.tests.server.tracking.test_tracker import DT, run


def still(n=60, x=20.0, y=0.0):
    tr = Tracker()
    run(tr, [[(x, y)]] * n)
    return tr


# ---- the formulas ------------------------------------------------------------------------------------------------------
def test_speed_and_heading_uncertainty_follow_from_the_velocity_covariance():
    speed, sigma, hsig = velocity_stats([3.0, 4.0], np.diag([0.04, 0.09]))
    assert speed == pytest.approx(5.0)
    assert sigma == pytest.approx(math.sqrt(0.36 * 0.04 + 0.64 * 0.09))               # u = (0.6, 0.8)
    assert hsig == pytest.approx(math.sqrt(0.64 * 0.04 + 0.36 * 0.09) / 5.0)         # perpendicular to the velocity, per metre/second
    speed, sigma, hsig = velocity_stats([0.0, 0.0], np.diag([0.04, 0.09]))
    assert speed == 0.0 and sigma == pytest.approx(0.3) and hsig is None


def test_the_zero_velocity_distance_is_the_mahalanobis_distance_of_the_velocity():
    assert zero_velocity_d2([1.0, 0.0], np.diag([0.25, 0.25])) == pytest.approx(4.0)
    assert zero_velocity_d2([0.0, 0.0], np.diag([0.25, 0.25])) == 0.0


# ---- outputs of tracks ---------------------------------------------------------------------------------------------------
def test_a_still_object_becomes_static_and_its_output_velocity_is_zero():
    o = still().outputs()[0]
    assert o.static and o.velocity_known and not o.velocity_out.any() and o.heading is None and o.heading_sigma is None
    assert o.hits == 60 and o.misses == 0 and o.existence == 1.0 and o.state == "confirmed"
    assert o.age_s == pytest.approx(5.9, abs=0.01) and o.age_scans == 60
    assert o.speed_sigma < OutputConfig().known_sigma_mps
    assert o.ellipse[0] >= o.ellipse[1] > 0 and o.cov.shape == (4, 4) and o.pos_cov.shape == (2, 2)


def test_a_young_track_is_not_static_because_its_velocity_is_not_known():
    o = still(4).outputs()[0]
    assert o.state == "confirmed" and not o.velocity_known and not o.static            # unknown velocity: able to move


def test_a_moving_object_is_dynamic_with_its_speed_and_heading():
    tr = Tracker()
    run(tr, [[(20.0, 5.0 * DT * k)] for k in range(60)])                                 # 5 m/s along +y
    o = tr.outputs()[0]
    assert not o.static and o.velocity_out == pytest.approx(o.velocity)
    assert o.speed == pytest.approx(5.0, abs=0.7) and o.heading == pytest.approx(math.pi / 2, abs=0.15)
    assert 0.0 < o.heading_sigma < 0.3 and o.speed_sigma > 0.0


def test_an_object_that_starts_to_move_leaves_the_static_state():
    tr = Tracker()
    run(tr, [[(20.0, 0.0)]] * 60)
    assert tr.outputs()[0].static
    run_more = [tr.step([_wc(20.0 + 3.0 * DT * (k + 1), 0.0, (60 + k) * DT)], (60 + k) * DT) for k in range(20)]
    assert len(run_more) == 20 and not tr.outputs()[0].static and tr.outputs()[0].speed > 1.0


def test_the_static_flag_needs_several_scans_in_a_row():
    cfg = OutputConfig(static_confirm_scans=5)
    tr = still(60)
    t = tr.tracks[0]
    t.static, t.static_count = False, 0
    from server.tracking.outputs import update_motion_state
    for k in range(4):
        update_motion_state(t, cfg)
        assert not t.static
    update_motion_state(t, cfg)
    assert t.static


def test_the_ellipse_is_the_95_percent_ellipse_of_the_position_covariance():
    tr = still(10)
    tr.tracks[0].filter.P[:2, :2] = np.diag([4.0, 1.0])
    major, minor, angle = tr.outputs()[0].ellipse
    assert major == pytest.approx(math.sqrt(5.99 * 4.0)) and minor == pytest.approx(math.sqrt(5.99))
    assert abs(math.sin(angle)) < 1e-6                                                    # the major axis lies along x


def test_tentative_tracks_have_no_output():
    tr = Tracker()
    run(tr, [[(20.0, 0.0)]])
    assert tr.outputs() == []


def _wc(x, y, t):
    from testing.tests.server.tracking.test_tracker import wc
    return wc(x, y, t)