"""The decision of a cycle and its causes (steps 7.5, 7.10 and 7.12)."""
import math
from dataclasses import replace

import pytest

from server.risk.decision import (NO_CAUSE, DecisionConfig, classify, decide, required_deceleration_to, respond_to_hazard)
from server.risk.hazards import Conflict
from server.risk.occlusion import VirtualHazard
from server.risk.risk_level import BUNDLES, RiskLevel
from server.risk.stopping import Regime, effective_distance, speed_caps

P = BUNDLES[RiskLevel.MEDIUM].apply()                                  # a_comf 2.5, a 6, margin 2, kappa 1.5, eps 0.02, delay 0.2 s


def conflict(tid=1, d=30.0, sigma=0.5, weight=1.0, state="confirmed", static=True, v_along=0.0, v_sigma=0.0, tau=0.0, existence=1.0):
    return Conflict(track_id=tid, state=state, static=static, tau_first_s=tau, s_conflict_m=d + 3.5, d_path_m=d, segment_m=(d + 3.5, d + 4.0),
                    t_clear_s=math.inf, v_along_mps=v_along, v_sigma_mps=v_sigma, sigma_d_m=sigma, margin_weight=weight, existence=existence)


def virtual(d, floor=2.0, tid=-2, kind="side_left"):
    return VirtualHazard(kind, d + 3.5, 0.8, 0.4, d, conflict(tid, d, sigma=0.0, weight=0.0, state="virtual", tau=0.4), floor)


def test_classify_uses_the_thresholds_of_the_parameters():
    assert classify(2.0, P) == (Regime.COMFORT, 0.0, False)
    assert classify(3.0, P) == (Regime.PROGRESSIVE, 3.0, False)
    assert classify(7.0, P) == (Regime.HARD, 7.0, False)
    assert classify(9.0, P) == (Regime.EMERGENCY, P.max_decel, True)
    assert classify(math.inf, P) == (Regime.EMERGENCY, P.max_decel, True)


def test_the_deceleration_to_a_target_speed():
    assert required_deceleration_to(2.0, 2.0, 5.0, 0.2) == 0.0 and required_deceleration_to(1.0, 2.0, 5.0, 0.2) == 0.0
    assert required_deceleration_to(10.0, 2.0, 20.0, 0.2) == pytest.approx(96.0 / 36.0)
    assert required_deceleration_to(10.0, 2.0, 2.0, 0.2) == math.inf


def test_without_hazards_the_caps_are_the_ceiling_and_there_is_no_cause():
    d = decide(1.0, 10.0, [], [], P)
    assert d.comfort_cap == d.safety_cap == DecisionConfig().ceiling_mps and d.comfort_cause == NO_CAUSE and d.brake_cause == NO_CAUSE
    assert d.regime == Regime.COMFORT and d.brake_request == 0.0 and not d.emergency and (d.n_hazards, d.n_virtual) == (0, 0)


def test_one_hazard_sets_both_caps_and_is_their_cause():
    d = decide(1.0, 10.0, [conflict(7)], [], P, risk_level=1)
    d_eff = effective_distance(30.0, P, 0.5, 0.0)
    comfort, safety = speed_caps(d_eff, P)
    assert d.comfort_cap == pytest.approx(comfort) and d.safety_cap == pytest.approx(safety)
    c = d.comfort_cause
    assert (c.kind, c.source_id, c.label, c.state) == ("track", 7, "track:7", "confirmed") and c.cap_mps == pytest.approx(comfort)
    assert c.d_path_m == 30.0 and c.d_eff_m == pytest.approx(d_eff) and d.safety_cause.label == "track:7" and d.risk_level == 1


def test_the_hazard_with_the_smallest_effective_distance_binds_even_if_it_is_farther():
    near, far = conflict(1, 25.0, sigma=0.0), conflict(2, 30.0, sigma=3.0)
    d = decide(1.0, 10.0, [near, far], [], P)
    assert d.comfort_cause.source_id == 2 and d.safety_cause.source_id == 2


def test_a_tentative_track_has_a_smaller_margin_and_a_virtual_one_none():
    caps = [decide(0.0, 10.0, [conflict(1, weight=w)], [], P).safety_cap for w in (1.0, 0.5, 0.0)]
    assert caps[0] < caps[1] < caps[2]
    assert caps[2] == pytest.approx(speed_caps(effective_distance(30.0, replace(P, margin_m=0.0), 0.5, 0.0), P)[1])


def test_a_lead_vehicle_is_allowed_a_higher_cap_than_a_standing_hazard():
    standing = decide(0.0, 10.0, [conflict(1)], [], P).safety_cap
    lead = decide(0.0, 10.0, [conflict(1, static=False, v_along=10.0, v_sigma=1.0)], [], P).safety_cap
    assert lead > standing


def test_a_hazard_beyond_the_ceiling_is_not_a_cause():
    d = decide(0.0, 10.0, [conflict(1, d=2000.0)], [], P)
    assert d.comfort_cap == d.safety_cap == 40.0 and d.comfort_cause == NO_CAUSE and d.n_hazards == 1


def test_a_virtual_hazard_has_the_floor_as_its_lowest_cap_and_is_named_by_its_place():
    v = virtual(0.0)
    d = decide(0.0, 1.5, [], [v], P)
    assert d.comfort_cap == d.safety_cap == 2.0
    assert (d.comfort_cause.kind, d.comfort_cause.source_id, d.comfort_cause.label) == ("virtual", -2, v.cause) and d.comfort_cause.label.startswith("occlusion:side_left@")
    assert d.regime == Regime.COMFORT and d.brake_request == 0.0 and not d.emergency and d.brake_cause == NO_CAUSE        # at the floor speed: allowed to pass


def test_a_car_faster_than_the_floor_slows_down_to_it_and_not_to_a_stop():
    v = virtual(15.0)
    d = decide(0.0, 14.0, [], [v], P)
    a_req = required_deceleration_to(14.0, 2.0, 15.0, P.delay_s)                                                          # margin 0 for a virtual hazard
    assert d.regime == Regime.HARD and d.brake_request == pytest.approx(a_req) and d.brake_cause.label == v.cause
    assert required_deceleration_to(14.0, 0.0, 15.0, P.delay_s) > a_req


def test_the_emergency_is_caused_by_the_hazard_that_cannot_be_avoided():
    d = decide(0.0, 20.0, [conflict(7, d=8.0), conflict(8, d=60.0)], [], P)
    assert d.emergency and d.regime == Regime.EMERGENCY and d.brake_request == P.max_decel
    assert d.brake_cause.label == "track:7" and d.brake_cause.regime == int(Regime.EMERGENCY)


def test_the_brake_cause_is_the_strongest_demand_and_the_cap_cause_can_be_another_hazard():
    far_uncertain, near = conflict(1, 40.0, sigma=6.0), conflict(2, 14.0, sigma=0.0)
    d = decide(0.0, 12.0, [far_uncertain, near], [], P)
    assert d.brake_cause.source_id == 2 and d.comfort_cause.source_id in (1, 2) and d.regime >= Regime.PROGRESSIVE


def test_the_cause_carries_what_the_evaluation_joins_with_the_ground_truth():
    d = decide(0.0, 10.0, [conflict(4, state="tentative", static=False, weight=0.5, tau=1.2, existence=0.4)], [], P)
    c = d.comfort_cause
    assert (c.state, c.static, c.tau_first_s, c.existence, c.margin_weight) == ("tentative", False, 1.2, 0.4, 0.5)
    assert set(c.to_dict()) >= {"kind", "source_id", "label", "state", "d_path_m", "d_eff_m", "cap_mps", "regime", "existence"}


def test_non_finite_numbers_are_none_in_the_dictionary():
    assert NO_CAUSE.to_dict()["d_path_m"] is None and NO_CAUSE.to_dict()["cap_mps"] is None and NO_CAUSE.to_dict()["kind"] == "none"


def test_respond_to_hazard_is_what_decide_uses():
    r = respond_to_hazard(conflict(3), 10.0, P)
    assert r.label == "track:3" and r.kind == "track" and r.floor_mps == 0.0
    d = decide(0.0, 10.0, [conflict(3)], [], P)
    assert d.responses[0].comfort_cap == r.comfort_cap