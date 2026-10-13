"""Collision probability by sampling and its deterministic fallback (step 7.7)."""
import itertools

import numpy as np
import pytest

from server.risk.collision import CollisionConfig, collision_probability
from server.risk.ego_path import EgoGeometry, EgoState, make_corridor
from server.risk.hazards import find_conflicts
from testing.tests.server.risk.test_hazards import COV, G, track

CFG = CollisionConfig(samples=300)


def corridor(v=10.0):
    return make_corridor(EgoState(v), G)


def prob(tracks, v=10.0, cfg=CFG, **kw):
    return collision_probability(tracks if isinstance(tracks, list) else [tracks], corridor(v), v, 0.0, G, cfg, **kw)


def test_no_hazards_no_probability():
    r = prob([])
    assert r.p_total == 0.0 and r.per_track == {} and r.method == "sampled"


def test_a_static_hazard_in_the_path_is_hit_in_every_sample():
    r = prob(track([30.0, -0.9], [31.8, 0.9]))
    assert r.p_total == pytest.approx(1.0) and r.method == "sampled" and r.sampled_ids == (1,)


def test_a_static_hazard_far_to_the_side_is_never_hit():
    assert prob(track([30.0, 12.0], [30.4, 12.4])).p_total == 0.0


def test_a_hazard_behind_the_car_is_never_hit():
    assert prob(track([-15.0, -0.9], [-11.0, 0.9])).p_total == 0.0


def test_a_standing_car_does_not_hit_what_is_far_ahead():
    assert prob(track([30.0, -0.9], [31.8, 0.9]), v=0.0).p_total == 0.0


def test_a_standing_car_is_hit_by_what_walks_into_it():
    ped = track([2.0, -3.2], [2.4, -2.8], vel=(0.0, 1.5), static=False)
    assert prob(ped, v=0.0).p_total > 0.9


def test_a_pedestrian_who_crosses_in_front_is_hit_and_one_who_walks_away_is_not():
    cross = track([20.0, -3.2], [20.4, -2.8], vel=(0.0, 1.5), static=False)
    away = track([20.0, -3.2], [20.4, -2.8], vel=(0.0, -1.5), static=False)
    assert prob(cross).p_total > 0.8 and prob(away).p_total < 0.1


def test_a_pedestrian_who_crosses_after_the_car_has_passed_is_not_hit():
    late = track([20.0, -9.0], [20.4, -8.6], vel=(0.0, 1.0), static=False)      # reaches the lane after 6.5 s, the car is past in 2.4 s
    assert prob(late).p_total < 0.05


def test_more_uncertainty_gives_more_probability_for_a_hazard_beside_the_path():
    beside = lambda cov: track([20.0, 2.5], [20.4, 2.9], static=True, cov=cov)          # 1.3 m from the corridor edge, standing still
    small, large = prob(beside(np.diag([0.01, 0.01, 0.01, 0.01]))).p_total, prob(beside(np.diag([1.0, 1.0, 0.25, 0.25]))).p_total
    assert small < 0.02 and large > 0.1


def test_the_result_is_reproducible_with_the_seed_and_changes_with_another():
    t = track([20.0, 2.5], [20.4, 2.9], static=False, cov=np.diag([1.0, 1.0, 0.25, 0.25]))
    a, b = prob(t), prob(t)
    c = prob(t, cfg=CollisionConfig(samples=300, seed=7))
    assert a.p_total == b.p_total and a.p_total != c.p_total


def test_the_total_combines_independent_tracks():
    t1 = track([20.0, 2.5], [20.4, 2.9], static=False, cov=np.diag([1.0, 1.0, 0.25, 0.25]), tid=1)
    t2 = track([25.0, -2.9], [25.4, -2.5], static=False, cov=np.diag([1.0, 1.0, 0.25, 0.25]), tid=2)
    r = prob([t1, t2])
    assert set(r.per_track) == {1, 2} and r.p_total == pytest.approx(1 - (1 - r.per_track[1]) * (1 - r.per_track[2]))
    assert max(r.per_track.values()) <= r.p_total <= 1.0


def test_the_deterministic_fallback_is_the_conflict_test():
    inside = track([30.0, -0.9], [31.8, 0.9], tid=1)
    side = track([30.0, 30.0], [30.4, 30.4], tid=2)
    r = prob([inside, side], cfg=CollisionConfig(force_deterministic=True), sigma_scale=1.5)
    assert r.method == "deterministic" and r.sampled_ids == () and r.per_track == {1: 1.0, 2: 0.0}
    ids = {c.track_id for c in find_conflicts([inside, side], corridor(), 10.0, 0.0, G, sigma_scale=1.5)}
    assert ids == {1}


def test_an_empty_time_budget_samples_the_nearest_track_and_answers_the_rest_deterministically():
    tracks = [track([30.0 + 5 * i, -0.9], [31.8 + 5 * i, 0.9], tid=i + 1) for i in range(3)]
    r = prob(tracks, cfg=CollisionConfig(samples=50, time_budget_s=0.0))
    assert r.method == "mixed" and r.sampled_ids == (1,) and r.per_track == {1: 1.0, 2: 1.0, 3: 1.0}


def test_the_budget_is_measured_with_the_clock():
    tracks = [track([30.0 + 5 * i, -0.9], [31.8 + 5 * i, 0.9], tid=i + 1) for i in range(3)]
    ticks = itertools.count(0.0, 0.005)
    r = prob(tracks, cfg=CollisionConfig(samples=50, time_budget_s=0.008), clock=lambda: next(ticks))
    assert r.sampled_ids == (1, 2) and r.method == "mixed"


def test_at_most_max_tracks_are_sampled():
    tracks = [track([30.0 + 3 * i, -0.9], [31.0 + 3 * i, 0.9], tid=i + 1) for i in range(5)]
    r = prob(tracks, cfg=CollisionConfig(samples=50, max_tracks=2))
    assert len(r.sampled_ids) == 2 and len(r.per_track) == 5


def test_the_sampling_fits_the_budget_for_a_typical_scene():
    tracks = [track([15.0 + 4 * i, -3.0 + i], [15.4 + 4 * i, -2.6 + i], vel=(0.0, 1.0), static=False, tid=i + 1) for i in range(6)]
    assert prob(tracks, cfg=CollisionConfig(samples=200, time_budget_s=1.0)).elapsed_s < 0.1       # generous: only a gross slowdown fails