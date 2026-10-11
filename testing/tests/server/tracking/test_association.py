"""Gate and Hungarian assignment (step 6.4)."""
import itertools

import numpy as np
import pytest

from server.tracking.association import AssociationConfig, associate, hungarian, mahalanobis_matrix
from server.tracking.kalman import ConstantVelocityFilter


def total(cost, pairs):
    return sum(cost[i, j] for i, j in pairs)


def brute_force(cost):
    n, m = cost.shape
    if n <= m:
        return min(sum(cost[i, p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))
    return min(sum(cost[p[j], j] for j in range(m)) for p in itertools.permutations(range(n), m))


# ---- the Hungarian algorithm ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(1, 1), (3, 3), (2, 5), (5, 2), (4, 6), (6, 4)])
def test_hungarian_finds_the_minimum_cost_assignment(shape):
    rng = np.random.default_rng(sum(shape))
    for _ in range(20):
        cost = rng.uniform(0, 10, shape)
        pairs = hungarian(cost)
        assert len(pairs) == min(shape)
        assert len({i for i, _ in pairs}) == len(pairs) and len({j for _, j in pairs}) == len(pairs)
        assert total(cost, pairs) == pytest.approx(brute_force(cost))


def test_hungarian_agrees_with_scipy_on_larger_problems():
    linear_sum_assignment = pytest.importorskip("scipy.optimize").linear_sum_assignment
    rng = np.random.default_rng(1)
    for shape in ((12, 12), (9, 15), (15, 9)):
        cost = rng.uniform(0, 100, shape)
        r, c = linear_sum_assignment(cost)
        assert total(cost, hungarian(cost)) == pytest.approx(cost[r, c].sum())


def test_hungarian_handles_empty_and_equal_costs():
    assert hungarian(np.zeros((0, 3))) == [] and hungarian(np.zeros((3, 0))) == []
    assert len(hungarian(np.ones((3, 3)))) == 3


def test_the_global_optimum_beats_greedy_nearest_neighbour():
    cost = np.array([[1.0, 2.0], [1.5, 9.0]])                     # greedy: 0-0 (1.0) then 1-1 (9.0); optimal: 0-1 and 1-0 (3.5)
    assert sorted(hungarian(cost)) == [(0, 1), (1, 0)]


# ---- gate and association -------------------------------------------------------------------------------------------------
def track(x, y, vx=0.0, vy=0.0, t=0.0, sigma=0.3):
    f = ConstantVelocityFilter.initialise([x, y], sigma ** 2 * np.eye(2), t)
    f.x[2:] = [vx, vy]
    f.P[2, 2] = f.P[3, 3] = 0.25
    return f


def meas(x, y, t=0.0, sigma=0.3):
    return (np.array([x, y]), sigma ** 2 * np.eye(2), t)


def test_each_measurement_goes_to_the_nearby_track():
    res = associate([track(10, 0), track(10, 5)], [meas(10.1, 5.1), meas(9.9, -0.1)])
    assert sorted(res.matches) == [(0, 1), (1, 0)] and not res.unmatched_tracks and not res.unmatched_measurements


def test_a_far_measurement_is_gated_and_left_over():
    res = associate([track(10, 0)], [meas(10.1, 0.0), meas(30.0, 5.0)])
    assert res.matches == [(0, 0)] and res.unmatched_measurements == [1] and res.unmatched_tracks == []


def test_only_one_measurement_per_track_and_one_track_per_measurement():
    res = associate([track(10, 0)], [meas(10.1, 0.0), meas(9.9, 0.1)])
    assert len(res.matches) == 1 and len(res.unmatched_measurements) == 1
    res = associate([track(10, 0), track(10.2, 0)], [meas(10.1, 0.0)])
    assert len(res.matches) == 1 and len(res.unmatched_tracks) == 1


def test_the_gate_uses_the_covariance_not_the_distance():
    f = track(20, 0)
    wide = (np.array([22.0, 0.0]), np.diag([4.0 ** 2, 0.1 ** 2]), 0.0)         # 2 m along a 4 m uncertainty: inside
    narrow = (np.array([20.0, 2.0]), np.diag([4.0 ** 2, 0.1 ** 2]), 0.0)       # 2 m across a 0.1 m uncertainty: far outside
    assert associate([f], [wide]).matches == [(0, 0)]
    assert associate([f], [narrow]).matches == []


def test_the_track_is_predicted_to_the_time_of_the_measurement():
    f = track(10, 0, vx=5.0, t=0.0)
    assert f.gate_distance2(*meas(10.5, 0.0, t=0.1)[:2], 0.1) < 1.0           # it moved 0.5 m
    assert associate([f], [meas(10.5, 0.0, t=0.1)]).matches == [(0, 0)]
    assert mahalanobis_matrix([f], [meas(10.5, 0.0, t=0.0)])[0, 0] > mahalanobis_matrix([f], [meas(10.5, 0.0, t=0.1)])[0, 0]


def test_the_filters_are_not_changed_and_empty_inputs_work():
    f = track(10, 0)
    x, p, t = f.x.copy(), f.P.copy(), f.t
    associate([f], [meas(10.1, 0.0, t=0.1)])
    assert f.x == pytest.approx(x) and f.P == pytest.approx(p) and f.t == t
    assert associate([], [meas(1, 1)]).unmatched_measurements == [0]
    assert associate([track(1, 1)], []).unmatched_tracks == [0]
    assert associate([], []).matches == []


def test_a_tighter_gate_rejects_what_a_wider_one_accepts():
    f, m = track(10, 0), meas(10.8, 0.0)                                       # about 1.9 sigma away
    d2 = associate([f], [m]).d2[0, 0]
    assert associate([f], [m], AssociationConfig(gate_chi2=d2 + 0.1)).matches == [(0, 0)]
    assert associate([f], [m], AssociationConfig(gate_chi2=d2 - 0.1)).matches == []