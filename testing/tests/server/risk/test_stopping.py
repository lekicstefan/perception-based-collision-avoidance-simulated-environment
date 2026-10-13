"""Stopping model, caps, required deceleration and regimes (steps 7.4 and 7.5), against the known answers of the specification (5.6)."""
import math

import pytest

from server.risk.stopping import (Regime, StoppingParams, effective_distance, impact_speed, lead_speed_low, required_deceleration,
                                  respond, speed_caps, stopping_distance, v_max, z_quantile)

P = StoppingParams(actuation_delay_s=0.25, jitter_margin_s=0.0, comfort_decel=3.0, assumed_decel=6.0, max_decel=8.0)     # t = 0.25 s


def test_worked_example_one_the_two_caps():
    v_comf, v_safe = speed_caps(30.0, P)
    assert v_safe == pytest.approx(17.5, abs=0.05) and v_comf == pytest.approx(12.7, abs=0.05)
    assert stopping_distance(v_safe, 0.25, 6.0) == pytest.approx(30.0)                          # the exact cap stops in exactly 30 m
    assert stopping_distance(17.5, 0.25, 6.0) == pytest.approx(30.0, abs=0.2)                   # the rounded 17.5 of the text: 4.4 + 25.5


def test_the_cap_is_the_inverse_of_the_stopping_distance():
    for a in (2.0, 3.0, 6.0, 8.0):
        for d in (1.0, 5.0, 30.0, 120.0):
            assert stopping_distance(v_max(a, 0.3, d), 0.3, a) == pytest.approx(d)
    assert v_max(6.0, 0.25, 0.0) == 0.0 and v_max(6.0, 0.25, -3.0) == 0.0


def test_the_comfort_cap_is_always_below_the_safety_cap():
    for d in (2.0, 10.0, 50.0):
        v_comf, v_safe = speed_caps(d, P)
        assert v_comf < v_safe


def test_worked_example_two_the_unavoidable_case():
    r = respond(16.7, 15.0, P)
    assert r.a_req == pytest.approx(12.9, abs=0.05)
    assert r.regime == Regime.EMERGENCY and r.emergency and r.brake_request == pytest.approx(8.0)
    assert r.impact_speed == pytest.approx(10.0, abs=0.5)                                       # about 37 km/h at impact
    assert impact_speed(16.7, 15.0, 0.25, 8.0) == pytest.approx(math.sqrt(16.7 ** 2 - 2 * 8.0 * (15.0 - 16.7 * 0.25)))


def test_emergency_when_the_delay_alone_uses_the_distance():
    r = respond(10.0, 2.0, P)                                                                    # d_eff <= v t = 2.5
    assert r.a_req == math.inf and r.regime == Regime.EMERGENCY and r.impact_speed == pytest.approx(10.0)


def test_required_deceleration_is_zero_when_standing():
    assert required_deceleration(0.0, 5.0, 0.25) == 0.0


def test_the_regimes_follow_the_required_deceleration():
    d = 40.0
    v_comf, v_safe = speed_caps(d, P)
    assert respond(v_comf * 0.9, d, P).regime == Regime.COMFORT
    assert respond(v_comf * 0.9, d, P).brake_request == 0.0
    mid = respond((v_comf + v_safe) / 2, d, P)
    assert mid.regime == Regime.PROGRESSIVE and 3.0 < mid.a_req <= 6.0 and mid.brake_request == pytest.approx(mid.a_req)
    v_8 = v_max(8.0, 0.25, d)
    hard = respond((v_safe + v_8) / 2, d, P)
    assert hard.regime == Regime.HARD and 6.0 < hard.a_req <= 8.0 and not hard.emergency
    assert respond(v_8 * 1.1, d, P).regime == Regime.EMERGENCY


def test_v_above_a_cap_is_the_same_as_a_required_deceleration_above_its_deceleration():
    d = 25.0
    for a in (3.0, 6.0, 8.0):
        v = v_max(a, 0.25, d)
        assert required_deceleration(v, d, 0.25) == pytest.approx(a)
        assert required_deceleration(v * 1.01, d, 0.25) > a > required_deceleration(v * 0.99, d, 0.25)


def test_the_effective_distance_of_a_static_hazard():
    p = StoppingParams(margin_m=2.0, kappa=1.5, epsilon=0.02)
    assert effective_distance(30.0, p, sigma_d=0.5) == pytest.approx(30.0 - 2.0 - z_quantile(0.02) * 1.5 * 0.5)


def test_a_lead_vehicle_gives_an_allowance_that_a_noisy_track_does_not():
    p = StoppingParams(hazard_decel=7.0, epsilon=0.10)
    assert effective_distance(30.0, p, v_h_low=10.0) == pytest.approx(30.0 + 100.0 / 14.0)
    assert lead_speed_low(10.0, 0.0, p) == pytest.approx(10.0)
    assert lead_speed_low(10.0, 2.0, p) == pytest.approx(10.0 - z_quantile(0.10) * 2.0)
    assert lead_speed_low(1.0, 2.0, p) == 0.0                                                    # a young track: no allowance


def test_a_larger_hazard_deceleration_gives_a_smaller_allowance():
    low, high = StoppingParams(hazard_decel=6.0), StoppingParams(hazard_decel=8.0)
    assert effective_distance(30.0, high, v_h_low=8.0) < effective_distance(30.0, low, v_h_low=8.0)


def test_the_quantile_of_the_acceptable_risk():
    assert z_quantile(0.10) == pytest.approx(1.2816, abs=1e-3)
    assert z_quantile(0.02) == pytest.approx(2.0537, abs=1e-3)
    assert z_quantile(0.001) == pytest.approx(3.0902, abs=1e-3)
    with pytest.raises(ValueError):
        z_quantile(0.0)


def test_the_delay_is_the_actuation_delay_plus_the_jitter_margin_and_nothing_else():
    assert StoppingParams(actuation_delay_s=0.2, jitter_margin_s=0.07).delay_s == pytest.approx(0.27)


def test_a_negative_effective_distance_gives_zero_caps_and_an_emergency():
    r = respond(5.0, -1.0, P)
    assert r.comfort_cap == 0.0 and r.safety_cap == 0.0 and r.regime == Regime.EMERGENCY


# Full emergency braking through the command executor from steady speed, measured in the simulator (docs: simulator_checks.md,
# data testing/tools/braking/stopping_table.csv): (km/h, stopping distance in m, time to stop in s).
STOPPING_TABLE = [(20, 2.759, 0.85), (30, 5.587, 1.20), (40, 9.379, 1.54), (50, 14.136, 1.89), (60, 19.858, 2.24),
                  (70, 26.544, 2.59), (80, 34.194, 2.93), (90, 42.809, 3.28), (100, 52.389, 3.63)]


def test_the_model_reproduces_the_measured_stopping_table_of_the_car():
    p = StoppingParams()
    assert p.actuation_delay_s == pytest.approx(0.15) and p.max_decel == pytest.approx(8.0)
    for kmh, dist, time in STOPPING_TABLE:
        v = kmh / 3.6
        assert stopping_distance(v, p.actuation_delay_s, p.max_decel) == pytest.approx(dist, abs=0.01)        # within 1 cm
        assert p.actuation_delay_s + v / p.max_decel == pytest.approx(time, abs=0.01)                          # the time is rounded to 0.01 s