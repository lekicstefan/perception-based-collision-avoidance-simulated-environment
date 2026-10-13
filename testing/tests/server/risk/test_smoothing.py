"""Hysteresis and smoothing (step 7.11). Times are multiples of 0.25 s so that the comparisons are exact."""
import math

import pytest

from server.risk.risk_level import RiskConfig, RiskLevel, reference_distance
from server.risk.smoothing import BrakeFilter, CapSmoother, RiskLevelFilter, SmoothingConfig

CFG = SmoothingConfig()                                       # rise 2 m/s^2 (comfort), hold 0.5 s, ceiling 40 m/s
DT = 0.25


def run(smoother, raws, t0=0.0):
    return [smoother.update(t0 + i * DT, r) for i, r in enumerate(raws)]


def test_the_first_value_is_taken_as_it_is():
    s = CapSmoother(2.0).update(0.0, 20.0)
    assert (s.value, s.raw, s.limited) == (20.0, 20.0, False)


def test_a_lower_cap_is_taken_at_once():
    out = run(CapSmoother(2.0), [20.0, 10.0, 3.0])
    assert [o.value for o in out] == [20.0, 10.0, 3.0] and not any(o.limited for o in out)


def test_a_higher_cap_waits_for_the_hold_and_then_rises_at_the_rate():
    out = run(CapSmoother(2.0), [20.0, 10.0, 20.0, 20.0, 20.0, 20.0, 20.0])
    assert [o.value for o in out] == [20.0, 10.0, 10.0, 10.0, 10.5, 11.0, 11.5]            # held at t = 0.5 and 0.75, then 0.5 m/s per 0.25 s
    assert [o.limited for o in out] == [False, False, True, True, True, True, True]


def test_the_cap_reaches_the_raw_value_and_stops_there():
    out = run(CapSmoother(2.0, SmoothingConfig(rise_hold_s=0.0)), [5.0] + [6.0] * 4)
    assert [o.value for o in out] == [5.0, 5.5, 6.0, 6.0, 6.0] and not out[-1].limited


def test_a_flickering_hazard_does_not_lift_the_cap():
    out = run(CapSmoother(2.0), [10.0, 20.0] * 12)
    assert all(o.value == 10.0 for o in out)


def test_a_new_low_resets_the_hold():
    out = run(CapSmoother(2.0), [20.0, 10.0, 20.0, 20.0, 10.0, 20.0, 20.0])
    assert out[-1].value == 10.0 and out[-2].value == 10.0                                  # the low at the fifth step restarted the 0.5 s


def test_the_ceiling_replaces_an_infinite_cap_and_a_nan_is_zero():
    assert CapSmoother(2.0).update(0.0, math.inf).value == CFG.ceiling_mps
    s = CapSmoother(2.0)
    s.update(0.0, 15.0)
    assert s.update(0.25, float("nan")).value == 0.0


def test_a_repeated_or_older_time_does_not_move_the_cap():
    s = CapSmoother(2.0, SmoothingConfig(rise_hold_s=0.0))
    s.update(1.0, 5.0)
    assert s.update(1.0, 9.0).value == 5.0 and s.update(0.5, 9.0).value == 5.0


def test_reset_forgets_the_history():
    s = CapSmoother(2.0)
    s.update(0.0, 5.0)
    s.reset()
    assert s.update(1.0, 30.0).value == 30.0


def test_the_braking_request_rises_at_once_and_falls_at_the_release_rate():
    f = BrakeFilter()
    out = [f.update(i * DT, r, False)[0] for i, r in enumerate([0.0, 3.0, 5.0, 0.0, 0.0, 0.0])]
    assert out == [0.0, 3.0, 5.0, 3.0, 1.0, 0.0]                                            # 8 m/s^2 per s = 2 per 0.25 s


def test_the_emergency_latches_and_releases_after_the_clear_time():
    f = BrakeFilter()
    seq = [f.update(i * DT, 0.0, e) for i, e in enumerate([True, False, False, False, False, False])]
    assert [s[1] for s in seq] == [True, True, True, False, False, False]                   # released 0.5 s after the last request
    assert seq[0][0] == 8.0 and seq[1][0] == 8.0 and seq[2][0] == 8.0
    assert seq[3][0] == 6.0 and seq[4][0] == 4.0                                           # then it runs down


def test_a_new_emergency_during_the_release_restarts_it():
    f = BrakeFilter()
    flags = [f.update(i * DT, 0.0, e)[1] for i, e in enumerate([True, False, False, True, False, False, False])]
    assert flags == [True, True, True, True, True, True, False]


def test_the_emergency_latch_gives_the_maximum_even_if_the_raw_request_is_smaller():
    f = BrakeFilter()
    f.update(0.0, 8.0, True)
    assert f.update(DT, 2.0, False)[0] == 8.0


def level_run(flt, views, t0=0.0):
    return [flt.update(t0 + i * DT, *v)[0].level for i, v in enumerate(views)]


GOOD = (100.0, 0.0, 10.0)                                     # sight, unknown fraction, speed
REF = reference_distance(10.0)
BAD = (1.0 * REF, 0.0, 10.0)                                  # one stopping distance of sight: high risk


def test_the_level_goes_up_at_once():
    f = RiskLevelFilter()
    assert level_run(f, [GOOD, BAD]) == [RiskLevel.LOW, RiskLevel.HIGH]


def test_the_level_comes_down_one_step_at_a_time_after_the_hold():
    f = RiskLevelFilter()
    levels = level_run(f, [GOOD, BAD] + [GOOD] * 19)                                         # t = 0, 0.25, then a good view from 0.5 s
    by_t = {i * DT: l for i, l in enumerate(levels)}
    assert by_t[2.25] == RiskLevel.HIGH and by_t[2.5] == RiskLevel.MEDIUM                      # 2.0 s after the first good view
    assert by_t[4.5] == RiskLevel.MEDIUM and by_t[4.75] == RiskLevel.LOW


def test_a_view_just_above_the_threshold_does_not_lower_the_level():
    f = RiskLevelFilter()
    borderline = (1.6 * REF, 0.0, 10.0)                                                       # raw: medium; but not clearly better than high
    assert set(level_run(f, [BAD] + [borderline] * 40)) == {RiskLevel.HIGH}


def test_the_unknown_fraction_has_the_same_hysteresis():
    hold = [(100.0, 0.30, 10.0)] * 40                                                         # raw medium, not below 0.35 * 0.8 = 0.28
    assert set(level_run(RiskLevelFilter(), [(100.0, 0.40, 10.0)] + hold)) == {RiskLevel.HIGH}
    out = level_run(RiskLevelFilter(), [(100.0, 0.40, 10.0)] + [(100.0, 0.20, 10.0)] * 20)
    assert out[0] == RiskLevel.HIGH and out[-1] == RiskLevel.MEDIUM                          # 0.20 is clearly better than high but still medium


def test_a_flickering_view_keeps_the_higher_level():
    f = RiskLevelFilter()
    out = level_run(f, [BAD, GOOD] * 20)
    assert set(out) == {RiskLevel.HIGH}


def test_the_assessment_carries_the_bundle_of_the_filtered_level_and_the_raw_level_is_returned():
    f = RiskLevelFilter()
    f.update(0.0, *BAD)
    a, raw = f.update(DT, *GOOD)
    assert a.level == RiskLevel.HIGH and raw == RiskLevel.LOW and a.stopping_params().epsilon == 0.001


def test_a_switched_off_layer_stays_at_its_fixed_level():
    f = RiskLevelFilter(RiskConfig(enabled=False, fixed_level=RiskLevel.LOW))
    assert set(level_run(f, [BAD, GOOD, BAD])) == {RiskLevel.LOW}