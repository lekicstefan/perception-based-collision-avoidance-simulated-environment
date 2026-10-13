"""Hazard prediction, reachable set and the time-of-arrival conflict test (step 7.3)."""
import math

import numpy as np
import pytest

from server.risk.ego_path import EgoGeometry, EgoState, make_corridor
from server.risk.hazards import (HazardConfig, corridor_interval, densify, find_conflicts, footprint_offsets, is_hazard, margin_weight,
                                 reachable_radius)
from server.risk.timing import PredictedTrack

G = EgoGeometry()
CORRIDOR = make_corridor(EgoState(10.0), G)                         # straight, half width 1.2 m, front bumper 3.5 m ahead of the axle
CFG = HazardConfig()
COV = np.diag([0.04, 0.04, 0.01, 0.01])


def track(lo, hi, vel=(0.0, 0.0), static=True, state="confirmed", cov=COV, tid=1):
    """A hazard whose visible box is [lo, hi] (vehicle frame); the nearest point is the one nearest to the sensor at the origin."""
    lo, hi = np.array(lo, float), np.array(hi, float)
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    nearest = np.array([lo[0], min(max(0.0, lo[1]), hi[1])])
    centre = (lo + hi) / 2
    return PredictedTrack(id=tid, t=1.0, position=centre, velocity=np.array(vel, float), cov=cov, static=static, velocity_known=True,
                          state=state, existence=1.0, age_s=3.0, surface=np.vstack([corners, nearest]) - centre)


def conflicts(tracks, v=10.0, travel=0.0, **kw):
    return find_conflicts(tracks if isinstance(tracks, list) else [tracks], CORRIDOR, v, travel, G, CFG, **kw)


def test_a_static_hazard_in_the_corridor_is_a_conflict_at_its_nearest_surface():
    (c,) = conflicts(track([30.0, -0.9], [31.8, 0.9]))
    assert c.tau_first_s == 0.0 and c.static and c.v_along_mps == 0.0
    r0 = reachable_radius(track([30.0, -0.9], [31.8, 0.9]), 0.0, 1.0, CFG)
    assert c.s_conflict_m == pytest.approx(30.0 - r0) and c.d_path_m == pytest.approx(30.0 - r0 - 3.5)


def test_the_distance_counts_from_the_front_bumper_at_the_time_of_effect():
    near = conflicts(track([30.0, -0.9], [31.8, 0.9]), travel=0.0)[0].d_path_m
    later = conflicts(track([30.0, -0.9], [31.8, 0.9]), travel=0.55)[0].d_path_m
    assert near - later == pytest.approx(0.55)


def test_a_hazard_behind_the_car_is_no_conflict():
    assert conflicts(track([-12.0, -0.9], [-8.0, 0.9])) == []


def test_a_hazard_far_to_the_side_is_no_conflict_within_the_horizon():
    assert conflicts(track([30.0, 30.0], [30.5, 30.5])) == []                  # a_obj tau^2 / 2 = 16 m after 4 s: cannot reach 29 m


def test_a_hazard_beyond_the_looked_at_length_is_ignored():
    assert conflicts(track([150.0, -0.9], [151.0, 0.9])) == []


def test_a_pedestrian_beside_the_road_is_a_conflict_for_a_slow_car_and_none_for_a_fast_one():
    ped = track([30.0, 10.0], [30.4, 10.4])                                     # 10 m to the side, 30 m ahead, standing
    slow = conflicts(ped, v=8.0)
    assert len(slow) == 1 and slow[0].tau_first_s > 2.0                         # it can only be there after a few seconds
    assert slow[0].t_clear_s + CFG.time_margin_s >= slow[0].tau_first_s
    assert conflicts(ped, v=30.0) == []                                         # the car is past the segment before anything can arrive


def test_a_car_that_stands_still_never_clears():
    ahead = conflicts(track([8.0, -0.9], [9.8, 0.9]), v=0.0)
    assert len(ahead) == 1 and ahead[0].t_clear_s == math.inf
    assert conflicts(track([30.0, 30.0], [30.5, 30.5]), v=0.0) == []            # out of reach: still nothing


def test_a_lead_moving_away_has_a_speed_along_the_path():
    lead = track([25.0, -0.9], [29.5, 0.9], vel=(10.0, 0.0), static=False)
    (c,) = conflicts(lead, v=12.0)
    assert c.v_along_mps == pytest.approx(10.0) and c.tau_first_s == 0.0 and not c.static


def test_a_wide_object_across_the_corridor_is_found_by_its_edges():
    wall = track([25.0, -6.0], [25.3, 6.0])                                     # both corners are outside the corridor
    (c,) = conflicts(wall)
    assert c.tau_first_s == 0.0 and c.s_conflict_m < 25.0


def test_a_larger_uncertainty_brings_the_first_meeting_forward():
    off = track([30.0, 6.0], [30.4, 6.4])
    small = conflicts(off, v=8.0, sigma_scale=1.0)[0].tau_first_s
    big = conflicts(track([30.0, 6.0], [30.4, 6.4], cov=np.diag([1.0, 1.0, 0.25, 0.25])), v=8.0, sigma_scale=3.0)[0].tau_first_s
    assert big < small


def test_a_pedestrian_walking_into_the_corridor_is_a_conflict_that_a_static_one_is_not_for_a_fast_car():
    walking = track([30.0, 3.0], [30.4, 3.4], vel=(0.0, -1.5), static=False)
    still = track([30.0, 3.0], [30.4, 3.4])
    assert conflicts(walking, v=20.0)[0].tau_first_s <= conflicts(still, v=20.0)[0].tau_first_s


def test_tentative_tracks_are_hazards_with_a_lower_margin_weight():
    t, c = track([30.0, -0.9], [31.8, 0.9], state="tentative"), track([30.0, -0.9], [31.8, 0.9])
    assert is_hazard(t) and margin_weight(t) == pytest.approx(0.5) and margin_weight(c) == 1.0
    assert conflicts(t)[0].margin_weight == pytest.approx(0.5) and conflicts(t)[0].d_path_m == pytest.approx(conflicts(c)[0].d_path_m)
    assert not is_hazard(t, HazardConfig(include_tentative=False))
    assert conflicts([track([30.0, -0.9], [31.8, 0.9], state="coasting")])[0].margin_weight == 1.0


def test_conflicts_are_sorted_nearest_first_and_carry_the_uncertainty():
    far, near = track([40.0, -0.9], [41.8, 0.9], tid=1), track([20.0, -0.9], [21.8, 0.9], tid=2)
    cs = conflicts([far, near], ego_sigma_m=0.5)
    assert [c.track_id for c in cs] == [2, 1] and cs[0].sigma_d_m == pytest.approx(math.sqrt(0.04 + 0.25), abs=1e-6)


def test_the_corridor_interval_of_discs():
    assert corridor_interval(np.array([[20.0, 5.0]]), 2.0, CORRIDOR) is None                           # 3.8 m from the strip
    a, b = corridor_interval(np.array([[20.0, 2.2]]), 2.0, CORRIDOR)                                   # 1.0 m from the strip
    assert a == pytest.approx(20.0 - math.sqrt(3.0)) and b == pytest.approx(20.0 + math.sqrt(3.0))


def test_the_reachable_radius_grows_with_time_and_inflation():
    t = track([30.0, 6.0], [30.4, 6.4])
    r = [reachable_radius(t, tau, 1.0, CFG) for tau in (0.0, 1.0, 2.0, 4.0)]
    assert r == sorted(r) and r[3] > 0.2 + 16.0 and reachable_radius(t, 1.0, 2.0, CFG) > r[1]


def test_the_edges_of_the_box_are_sampled():
    pts = densify(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5]]), 0.25)
    assert len(pts) == 12 and np.hypot(*(pts[1] - pts[0])) == pytest.approx(0.25)
    assert footprint_offsets(track([30.0, -0.9], [31.8, 0.9]), CFG).shape[1] == 2
    bare = PredictedTrack(1, 1.0, np.zeros(2), np.zeros(2), COV, True, True, "confirmed", 1.0, 1.0)
    assert footprint_offsets(bare).shape == (1, 2)                                                      # no surface: the tracked point