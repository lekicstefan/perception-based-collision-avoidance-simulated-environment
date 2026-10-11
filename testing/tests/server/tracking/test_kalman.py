"""The constant-velocity Kalman filter (step 6.2): model matrices, convergence, consistency, time stamps."""
import math

import numpy as np
import pytest

from server.tracking.kalman import ConstantVelocityFilter, FilterConfig, H, process_noise, transition

I2 = np.eye(2)


def r_iso(sigma):
    return sigma ** 2 * I2


# ---- model -------------------------------------------------------------------------------------------------------------
def test_transition_moves_the_position_by_velocity_times_dt():
    f = transition(0.25)
    x = f @ np.array([1.0, 2.0, 4.0, -8.0])
    assert x == pytest.approx([2.0, 0.0, 4.0, -8.0])
    assert transition(0.0) == pytest.approx(np.eye(4))


def test_process_noise_is_the_white_noise_acceleration_model():
    q = process_noise(0.1, 2.0)
    assert q[0, 0] == pytest.approx(4.0 * 0.1 ** 4 / 4) and q[2, 2] == pytest.approx(4.0 * 0.1 ** 2)
    assert q[0, 2] == pytest.approx(4.0 * 0.1 ** 3 / 2) and q[1, 3] == q[0, 2] and q[0, 1] == 0.0
    assert q == pytest.approx(q.T)
    assert np.all(np.linalg.eigvalsh(q) > -1e-15)
    assert process_noise(0.0, 2.0) == pytest.approx(np.zeros((4, 4)))
    assert process_noise(0.2, 1.0)[2, 2] == pytest.approx(4 * process_noise(0.1, 1.0)[2, 2])      # velocity variance ~ dt^2
    assert process_noise(0.1, 2.0) == pytest.approx(4 * process_noise(0.1, 1.0))                  # ~ sigma_a^2


def test_a_new_track_has_the_measured_position_and_a_large_velocity_uncertainty():
    f = ConstantVelocityFilter.initialise([10.0, -3.0], r_iso(0.2), 1.5)
    assert f.position == pytest.approx([10.0, -3.0]) and f.velocity == pytest.approx([0.0, 0.0])
    assert f.P[0, 0] == pytest.approx(0.04) and f.P[2, 2] == pytest.approx(FilterConfig().initial_velocity_sigma ** 2)
    assert f.t == 1.5


def test_prediction_moves_the_track_and_grows_the_uncertainty_with_the_real_dt():
    f = ConstantVelocityFilter(np.array([0.0, 0.0, 5.0, 1.0]), np.diag([0.01, 0.01, 0.04, 0.04]), 0.0)
    f.predict(0.1)
    assert f.position == pytest.approx([0.5, 0.1]) and f.t == pytest.approx(0.1)
    g = ConstantVelocityFilter(np.array([0.0, 0.0, 5.0, 1.0]), np.diag([0.01, 0.01, 0.04, 0.04]), 0.0)
    g.predict(0.5)                                                       # a missed scan: five times the interval
    assert g.position == pytest.approx([2.5, 0.5])
    assert g.P[0, 0] > f.P[0, 0] and g.P[2, 2] > f.P[2, 2]
    with pytest.raises(ValueError):
        g.predict(0.4)                                                   # no stepping backwards by predict()


def test_predicted_does_not_change_the_track():
    f = ConstantVelocityFilter(np.array([0.0, 0.0, 5.0, 0.0]), np.eye(4), 1.0)
    x, p = f.predicted(1.3)
    assert x[:2] == pytest.approx([1.5, 0.0]) and f.t == 1.0 and f.position == pytest.approx([0.0, 0.0])
    xb, _ = f.predicted(0.9)                                             # also before the track time
    assert xb[:2] == pytest.approx([-0.5, 0.0])


# ---- update ------------------------------------------------------------------------------------------------------------
def test_an_update_pulls_the_state_towards_the_measurement_and_shrinks_the_covariance():
    f = ConstantVelocityFilter.initialise([0.0, 0.0], r_iso(0.5), 0.0)
    before = f.predicted(0.1)[1]
    res = f.update([0.4, 0.0], r_iso(0.1), 0.1)
    assert res.applied and res.reason == "ok" and res.dt == pytest.approx(0.1)
    assert 0.0 < f.position[0] <= 0.4 and f.velocity[0] > 0.0
    assert np.trace(f.P) < np.trace(before)
    assert f.P == pytest.approx(f.P.T) and np.all(np.linalg.eigvalsh(f.P) > 0)


def test_a_stationary_object_converges_to_zero_velocity():
    rng = np.random.default_rng(1)
    f = ConstantVelocityFilter.initialise([20.0, 3.0] + rng.normal(0, 0.1, 2), r_iso(0.1), 0.0)
    for k in range(1, 101):
        f.update([20.0, 3.0] + rng.normal(0, 0.1, 2), r_iso(0.1), 0.1 * k)
    assert np.linalg.norm(f.position - [20.0, 3.0]) < 0.25
    assert f.speed < 0.6


def test_a_moving_object_converges_to_its_velocity():
    rng = np.random.default_rng(2)
    truth = lambda t: np.array([30.0 - 5.0 * t, 2.0 + 1.0 * t])  # noqa: E731
    f = ConstantVelocityFilter.initialise(truth(0.0) + rng.normal(0, 0.1, 2), r_iso(0.1), 0.0)
    for k in range(1, 41):
        t = 0.1 * k
        f.update(truth(t) + rng.normal(0, 0.1, 2), r_iso(0.1), t)
    assert f.velocity == pytest.approx([-5.0, 1.0], abs=0.5)
    assert f.speed == pytest.approx(math.hypot(5.0, 1.0), abs=0.5)
    assert abs(math.degrees(f.heading) - math.degrees(math.atan2(1.0, -5.0))) < 5.0


def test_the_filter_is_consistent_when_the_model_is_true():
    """A target that really follows the model (random acceleration of standard deviation sigma_a, noisy positions): the
    normalised innovation squared of a 2-dimensional measurement averages 2."""
    rng = np.random.default_rng(7)
    cfg = FilterConfig(sigma_a=1.0)
    dt, sigma_m = 0.1, 0.2
    x = np.array([0.0, 0.0, 3.0, -1.0])
    f = ConstantVelocityFilter.initialise(x[:2] + rng.normal(0, sigma_m, 2), r_iso(sigma_m), 0.0, cfg)
    nis = []
    g = np.array([dt ** 2 / 2, dt])
    for k in range(1, 3001):
        a = rng.normal(0, cfg.sigma_a, 2)
        x = transition(dt) @ x
        x[:2] += g[0] * a
        x[2:] += g[1] * a
        res = f.update(H @ x + rng.normal(0, sigma_m, 2), r_iso(sigma_m), dt * k)
        if k > 30:
            nis.append(res.nis)
    assert 1.75 < np.mean(nis) < 2.3


def test_the_gate_distance_is_small_near_the_prediction_and_large_far_away():
    f = ConstantVelocityFilter(np.array([10.0, 0.0, 2.0, 0.0]), np.diag([0.1, 0.1, 0.1, 0.1]), 0.0)
    near = f.gate_distance2([10.2, 0.0], r_iso(0.2), 0.1)
    far = f.gate_distance2([14.0, 3.0], r_iso(0.2), 0.1)
    assert near < 1.0 < 20.0 < far
    assert f.t == 0.0                                                                   # nothing changed


# ---- time stamps -------------------------------------------------------------------------------------------------------
def test_two_measurements_with_the_same_time_stamp_are_two_updates_without_a_prediction_between():
    f = ConstantVelocityFilter.initialise([5.0, 0.0], r_iso(0.3), 1.0)
    first = f.update([5.1, 0.0], r_iso(0.3), 1.0)
    p_one = f.P.copy()
    second = f.update([4.9, 0.0], r_iso(0.3), 1.0)
    assert first.reason == second.reason == "same_time" and second.applied
    assert f.t == 1.0 and np.trace(f.P) < np.trace(p_one)                               # no process noise was added


def test_an_out_of_order_measurement_is_used_without_moving_the_track_time():
    f = ConstantVelocityFilter.initialise([0.0, 0.0], r_iso(0.1), 0.0)
    f.update([0.5, 0.0], r_iso(0.1), 0.1)
    f.update([1.0, 0.0], r_iso(0.1), 0.2)
    t_before = f.t
    res = f.update([0.52, 0.0], r_iso(0.1), 0.1)                                        # late by 0.1 s
    assert res.applied and res.reason == "out_of_order" and res.dt == pytest.approx(-0.1)
    assert f.t == t_before and f.position[0] == pytest.approx(1.0, abs=0.15)
    assert f.P == pytest.approx(f.P.T) and np.all(np.linalg.eigvalsh(f.P) > 0)


def test_out_of_order_and_in_order_give_nearly_the_same_estimate():
    ordered = ConstantVelocityFilter.initialise([0.0, 0.0], r_iso(0.1), 0.0)
    ordered.update([0.5, 0.0], r_iso(0.1), 0.1)
    ordered.update([1.0, 0.0], r_iso(0.1), 0.2)
    swapped = ConstantVelocityFilter.initialise([0.0, 0.0], r_iso(0.1), 0.0)
    swapped.update([1.0, 0.0], r_iso(0.1), 0.2)
    swapped.update([0.5, 0.0], r_iso(0.1), 0.1)
    assert swapped.position == pytest.approx(ordered.position, abs=0.15)
    assert swapped.velocity[0] == pytest.approx(ordered.velocity[0], abs=2.0)


def test_a_measurement_that_is_too_old_is_rejected_and_changes_nothing():
    f = ConstantVelocityFilter.initialise([0.0, 0.0], r_iso(0.1), 2.0)
    x, p = f.x.copy(), f.P.copy()
    res = f.update([9.0, 9.0], r_iso(0.1), 1.0)                                         # one second late
    assert not res.applied and res.reason == "too_old"
    assert f.x == pytest.approx(x) and f.P == pytest.approx(p) and f.t == 2.0


def test_the_uncertainty_grows_while_a_track_coasts_and_the_filter_keeps_working_at_a_low_rate():
    f = ConstantVelocityFilter.initialise([0.0, 0.0], r_iso(0.1), 0.0)
    for k in range(1, 6):
        f.update([1.0 * k, 0.0], r_iso(0.1), 0.2 * k)
    p0 = f.P[0, 0]
    f.predict(f.t + 1.0)                                                                # five scans missed
    assert f.P[0, 0] > 5 * p0
    res = f.update(f.position + [0.1, 0.0], r_iso(0.1), f.t)                            # a measurement arrives again
    assert res.applied and res.nis < 10.0