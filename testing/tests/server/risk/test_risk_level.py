"""Risk level and the parameter bundles (step 7.8)."""
import pytest

from server.risk.risk_level import BUNDLES, RiskBundle, RiskConfig, RiskLevel, assess, reference_distance
from server.risk.stopping import StoppingParams, stopping_distance


def test_the_bundles_are_the_table_of_the_specification():
    low, mid, high = (BUNDLES[l] for l in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH))
    assert (low.epsilon, mid.epsilon, high.epsilon) == (0.10, 0.02, 0.001)
    assert (low.comfort_decel, mid.comfort_decel, high.comfort_decel) == (3.0, 2.5, 2.0)
    assert (low.assumed_decel, mid.assumed_decel, high.assumed_decel) == (6.5, 6.0, 5.0)
    assert (low.margin_m, mid.margin_m, high.margin_m) == (1.0, 2.0, 3.0)
    assert (low.kappa, mid.kappa, high.kappa) == (1.0, 1.5, 2.0)
    assert (low.occluded_speed_mps, mid.occluded_speed_mps, high.occluded_speed_mps) == (1.5, 2.0, 2.5)
    assert (low.hazard_decel, mid.hazard_decel, high.hazard_decel) == (6.0, 7.0, 8.0)


def test_a_bundle_replaces_the_whole_set_and_keeps_the_delay_and_the_real_capability():
    base = StoppingParams(actuation_delay_s=0.2, jitter_margin_s=0.07, max_decel=8.5)
    p = BUNDLES[RiskLevel.HIGH].apply(base)
    assert (p.epsilon, p.comfort_decel, p.assumed_decel, p.margin_m, p.kappa, p.hazard_decel) == (0.001, 2.0, 5.0, 3.0, 2.0, 8.0)
    assert (p.actuation_delay_s, p.jitter_margin_s, p.max_decel) == (0.2, 0.07, 8.5)


def test_a_higher_level_gives_lower_caps():
    from server.risk.stopping import speed_caps
    caps = [speed_caps(30.0, BUNDLES[l].apply()) for l in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH)]
    assert caps[0][0] > caps[1][0] > caps[2][0] and caps[0][1] > caps[1][1] > caps[2][1]


def test_a_long_free_view_is_low_risk():
    a = assess(60.0, 0.0, 10.0)
    assert a.level == RiskLevel.LOW and a.cause == "clear" and a.bundle == BUNDLES[RiskLevel.LOW]


def test_the_sight_is_judged_in_stopping_distances():
    ref = reference_distance(15.0)
    assert ref == pytest.approx(stopping_distance(15.0, StoppingParams().delay_s, 6.0))
    assert assess(2.0 * ref, 0.0, 15.0).level == RiskLevel.MEDIUM
    assert assess(1.0 * ref, 0.0, 15.0).level == RiskLevel.HIGH and assess(1.0 * ref, 0.0, 15.0).cause == "sight"
    assert assess(4.0 * ref, 0.0, 15.0).level == RiskLevel.LOW
    assert assess(30.0, 0.0, 8.0).level == RiskLevel.LOW and assess(30.0, 0.0, 20.0).level == RiskLevel.HIGH      # same view, other speed


def test_a_standing_or_slow_car_is_compared_with_the_minimum_distance():
    assert reference_distance(0.0) == 10.0 == reference_distance(2.0)
    assert assess(12.0, 0.0, 0.0).level == RiskLevel.HIGH and assess(40.0, 0.0, 0.0).level == RiskLevel.LOW


def test_the_unknown_fraction_raises_the_level():
    assert assess(100.0, 0.10, 10.0).level == RiskLevel.LOW
    a = assess(100.0, 0.20, 10.0)
    assert a.level == RiskLevel.MEDIUM and a.cause == "unknown"
    assert assess(100.0, 0.40, 10.0).level == RiskLevel.HIGH


def test_the_larger_of_the_two_criteria_wins():
    a = assess(2.0 * reference_distance(10.0), 0.40, 10.0)
    assert a.level == RiskLevel.HIGH and a.cause == "unknown"
    b = assess(1.0 * reference_distance(10.0), 0.40, 10.0)
    assert b.level == RiskLevel.HIGH and b.cause == "both"


def test_the_layer_can_be_switched_off():
    off = RiskConfig(enabled=False)
    a = assess(5.0, 0.9, 20.0, off)
    assert a.level == RiskLevel.MEDIUM and a.cause == "off" and a.bundle == BUNDLES[RiskLevel.MEDIUM]
    assert assess(5.0, 0.9, 20.0, RiskConfig(enabled=False, fixed_level=RiskLevel.LOW)).bundle == BUNDLES[RiskLevel.LOW]


def test_the_assessment_gives_the_stopping_parameters():
    a = assess(5.0, 0.9, 20.0)
    assert a.level == RiskLevel.HIGH and a.stopping_params().epsilon == 0.001
    assert isinstance(a.bundle, RiskBundle)