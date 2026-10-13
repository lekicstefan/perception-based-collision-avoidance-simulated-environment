"""Sensitivity analysis of the speed caps (step 7.9): the direction and the size of every effect, and the tool."""
import math

import pytest

from server.risk.risk_level import BUNDLES, RiskLevel
from testing.support import risk_sensitivity as rs


def col(rows, i):
    return [r[i] for r in rows]


def nondecreasing(xs):
    return all(b >= a - 1e-9 for a, b in zip(xs, xs[1:]))


def nonincreasing(xs):
    return all(b <= a + 1e-9 for a, b in zip(xs, xs[1:]))


def test_a_larger_acceptable_risk_raises_both_caps():
    s = rs.sweep("epsilon")
    assert nondecreasing(col(s, 1)) and nondecreasing(col(s, 2))
    assert 0.2 < col(s, 2)[-1] - col(s, 2)[0] < 1.5                      # the specification's remark: the quantile alone moves the cap very little


def test_more_inflation_and_margin_lower_both_caps():
    for name, least in (("kappa", 0.5), ("margin", 1.0)):
        s = rs.sweep(name)
        assert nonincreasing(col(s, 1)) and nonincreasing(col(s, 2)) and col(s, 2)[0] - col(s, 2)[-1] > least


def test_the_comfort_deceleration_moves_the_comfort_cap_only_and_the_assumed_one_the_safety_cap_only():
    c, a = rs.sweep("comfort_decel"), rs.sweep("assumed_decel")
    assert nondecreasing(col(c, 1)) and max(col(c, 2)) - min(col(c, 2)) < 1e-9
    assert nondecreasing(col(a, 2)) and max(col(a, 1)) - min(col(a, 1)) < 1e-9


def test_the_hazard_braking_matters_for_a_lead_and_not_for_a_hazard_in_the_path():
    lead, static = rs.sweep("hazard_decel", rs.LEAD), rs.sweep("hazard_decel", rs.STATIC)
    assert nonincreasing(col(lead, 2)) and col(lead, 2)[0] - col(lead, 2)[-1] > 0.3
    assert max(col(static, 2)) - min(col(static, 2)) < 1e-9


def test_a_longer_delay_lowers_the_caps():
    s = rs.sweep("delay")
    assert nonincreasing(col(s, 1)) and nonincreasing(col(s, 2)) and col(s, 2)[0] - col(s, 2)[-1] > 0.5


def test_the_levels_are_ordered_at_every_distance():
    for scene in (rs.STATIC, rs.LEAD):
        c = rs.level_curves([5, 15, 30, 60], scene)
        for i in range(4):
            assert c["low"][i][1] > c["medium"][i][1] > c["high"][i][1] and c["low"][i][2] > c["medium"][i][2] > c["high"][i][2]


def test_the_risk_layer_has_a_measurable_effect():
    e = rs.level_effect(rs.STATIC)
    assert e["drop_safety"] > 0.10 and e["drop_comfort"] > 0.15                    # at 30 m low -> high risk lowers the caps by more than a tenth
    assert e["low"][1] - e["high"][1] > 2.0


def test_the_tornado_is_sorted_and_every_parameter_acts_in_the_expected_direction():
    rows = rs.tornado(rs.LEAD)
    assert [r[0] for r in rows] and len(rows) == 6
    sizes = [max(abs(r[3]), abs(r[4])) for r in rows]
    assert sizes == sorted(sizes, reverse=True)
    by = {r[0]: r for r in rows}
    assert by["epsilon"][4] < 0 and by["kappa"][4] < 0 and by["margin"][4] < 0 and by["hazard_decel"][4] < 0
    assert by["comfort_decel"][3] < 0 and by["comfort_decel"][4] == pytest.approx(0.0, abs=1e-9)
    assert by["assumed_decel"][4] < 0 and by["assumed_decel"][3] == pytest.approx(0.0, abs=1e-9)


def test_the_double_counted_covariance_is_the_quantile_times_kappa_times_sigma():
    rows = rs.double_count((0.5,))
    med = next(r for r in rows if r[0] == "medium")
    z = BUNDLES[RiskLevel.MEDIUM].apply().z
    assert med[2] == pytest.approx(z * 1.5 * 0.5) and med[3] > med[4] > 0.0
    assert all(r[3] >= r[4] for r in rows)


def test_a_virtual_side_hazard_never_stops_the_car():
    for lvl in RiskLevel:
        assert rs.virtual_cap(lvl, 0.0) == BUNDLES[lvl].occluded_speed_mps and rs.virtual_cap(lvl, 40.0) > rs.virtual_cap(lvl, 5.0) > 0.0


def test_the_occluded_approach_speed_hardly_changes_the_conflict_test():
    assert math.isinf(rs.no_conflict_speed(0.8, 2.5, 20.0)) and math.isinf(rs.no_conflict_speed(0.5, 1.5, 20.0))     # tau <= the 0.5 s margin: never
    assert rs.no_conflict_speed(0.8, 1.5, 20.0) > 100.0 and rs.no_conflict_speed(2.5, 1.5, 20.0) > 15.0              # otherwise an absurd speed


def test_the_tool_writes_its_files(tmp_path):
    from testing.tools.risk.sensitivity_check import main
    out = main(["--out", str(tmp_path)])
    for name in ("summary.txt", "sweeps.png", "levels.png", "tornado.png", "tornado.csv", "double_count.csv", "sweep_epsilon_static.csv"):
        assert (out / name).stat().st_size > 0
    assert "low -> high" in (out / "summary.txt").read_text()


def test_the_deceleration_assumptions_are_the_strongest_levers_of_the_bundle():
    big = {r[0]: max(abs(r[3]), abs(r[4])) for r in rs.tornado(rs.STATIC)}
    assert big["assumed_decel"] > 3 * big["epsilon"] and big["comfort_decel"] > 3 * big["epsilon"] and big["assumed_decel"] > big["margin"]