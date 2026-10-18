"""A conflict that exists only because an object could reach the corridor is answered with a floor speed (step 7.15 follow-up):
a car parked beside the road is passed, not stopped for. An object whose nominal motion enters the corridor is not affected."""
import math
from dataclasses import replace

import numpy as np
import pytest

from server.risk.decision import DecisionConfig, decide
from server.risk.ego_path import EgoGeometry, EgoState, make_corridor
from server.risk.hazards import Conflict, find_conflicts
from server.risk.occlusion import VirtualHazard
from server.risk.risk_level import BUNDLES, RiskLevel
from server.risk.timing import PredictedTrack
from server.risk.stopping import Regime, lead_speed_low

P = BUNDLES[RiskLevel.MEDIUM].apply()
FLOOR = BUNDLES[RiskLevel.MEDIUM].occluded_speed_mps                   # 2.0
PASS = 40.0 / 3.6                                                       # the passing speed of the live system, 11.1 m/s
G = EgoGeometry()


def track(x, y, vx=0.0, vy=0.0, static=True):
    return PredictedTrack(id=1, t=0.0, position=np.array([x, y]), velocity=np.array([vx, vy]), cov=np.diag([0.04, 0.04, 0.01, 0.01]), static=static,
                          velocity_known=True, state="confirmed", existence=1.0, age_s=5.0)


def conflicts(tr, speed=2.0):
    return find_conflicts([tr], make_corridor(EgoState(speed), G), speed, 0.0, G, sigma_scale=1.0)


def test_a_parked_object_beside_the_corridor_is_a_precaution_only():
    (c,) = conflicts(track(30.0, 3.5))
    assert not c.in_corridor


def test_an_object_in_the_lane_is_in_the_corridor():
    (c,) = conflicts(track(30.0, 0.0))
    assert c.in_corridor


def test_an_object_walking_into_the_lane_is_in_the_corridor_and_one_walking_away_is_not():
    (into,) = conflicts(track(30.0, 3.5, 0.0, -1.5, static=False))
    assert into.in_corridor
    (away,) = conflicts(track(30.0, 3.5, 0.0, 1.5, static=False))
    assert not away.in_corridor


def test_a_velocity_that_is_only_noise_is_not_walking_into_the_lane():
    (noisy,) = conflicts(track(30.0, 3.5, 0.0, -0.5, static=False))                  # 0.5 m/s with a standard deviation of 0.1: 0.3 m/s significant
    assert not noisy.in_corridor
    (walking,) = conflicts(track(30.0, 3.5, 0.0, -1.0, static=False))                # 1 m/s: 0.8 m/s significant, 3.2 m in the horizon
    assert walking.in_corridor


def conflict(d, in_corridor):
    return Conflict(track_id=1, state="confirmed", static=True, tau_first_s=0.5, s_conflict_m=d + 3.5, d_path_m=d, segment_m=(d + 3.5, d + 4.0),
                    t_clear_s=10.0, v_along_mps=0.0, v_sigma_mps=0.0, sigma_d_m=0.5, margin_weight=1.0, existence=1.0, in_corridor=in_corridor)


def test_the_precaution_lets_the_car_pass_at_the_floor_speed():
    d = decide(0.0, 1.5, [conflict(3.3, False)], [], P, side_floor_mps=FLOOR)                  # the margin is used up, the car is slower than the floor
    assert d.comfort_cap == d.safety_cap == FLOOR and d.brake_request == 0.0 and not d.emergency and d.regime == Regime.COMFORT


def test_a_faster_car_slows_to_the_floor_and_not_to_a_stop():
    d = decide(0.0, 10.0, [conflict(40.0, False)], [], P, side_floor_mps=FLOOR)
    assert d.comfort_cap >= FLOOR and d.safety_cap >= FLOOR
    stop = decide(0.0, 10.0, [conflict(3.3, True)], [], P, side_floor_mps=FLOOR)
    assert stop.brake_request == P.max_decel                                                   # a real obstacle in the lane is still stopped for


def test_an_object_in_the_corridor_gets_no_floor():
    d = decide(0.0, 1.5, [conflict(3.3, True)], [], P, side_floor_mps=FLOOR)
    assert d.safety_cap < FLOOR and d.brake_request == P.max_decel


def test_the_floor_can_be_switched_off():
    d = decide(0.0, 1.5, [conflict(3.3, False)], [], P, cfg=DecisionConfig(side_floor_enabled=False), side_floor_mps=FLOOR)
    assert d.safety_cap < FLOOR

def walking(d, v, sigma=0.0, age=math.inf):
    """A precaution-only conflict whose hazard moves ahead along the path at v m/s (standard deviation sigma) and whose track is `age` seconds old."""
    return Conflict(track_id=1, state="confirmed", static=False, tau_first_s=0.5, s_conflict_m=d + 3.5, d_path_m=d, segment_m=(d + 3.5, d + 4.0),
                    t_clear_s=10.0, v_along_mps=v, v_sigma_mps=sigma, sigma_d_m=0.5, margin_weight=1.0, existence=1.0, in_corridor=False, age_s=age)


def test_the_floor_is_relative_to_a_hazard_that_moves_ahead_along_the_path():
    d = decide(0.0, 1.5, [walking(3.3, 1.4)], [], P, side_floor_mps=FLOOR)
    assert d.comfort_cap == pytest.approx(FLOOR + lead_speed_low(1.4, 0.0, P))


def test_a_car_passing_at_the_pace_of_a_walker_is_not_braked_for():
    d = decide(0.0, 3.0, [walking(3.3, 1.4)], [], P, side_floor_mps=FLOOR)       # faster than the old absolute floor (2.0), slower than the new one (3.4)
    assert d.brake_request == 0.0 and d.comfort_cap >= 3.0


def test_the_lower_quantile_of_the_speed_is_used_so_that_a_noisy_velocity_gives_less():
    d = decide(0.0, 1.5, [walking(3.3, 1.4, sigma=0.35, age=20.0)], [], P, side_floor_mps=FLOOR)       # about what the walking pedestrian of the dev scene shows
    assert d.comfort_cap == pytest.approx(FLOOR + lead_speed_low(1.4, 0.35, P)) and FLOOR < d.comfort_cap < FLOOR + 1.4


def test_the_relative_floor_needs_an_established_track():
    old = decide(0.0, 1.5, [walking(3.3, 1.4, sigma=0.1, age=3.0)], [], P, side_floor_mps=FLOOR)
    assert old.comfort_cap == pytest.approx(FLOOR + lead_speed_low(1.4, 0.1, P))
    young = decide(0.0, 1.5, [walking(3.3, 1.4, sigma=0.1, age=2.9)], [], P, side_floor_mps=FLOOR)
    assert young.comfort_cap == pytest.approx(FLOOR)                              # just below the limit: the plain floor


def test_a_phantom_velocity_of_a_young_track_does_not_lift_the_floor():
    d = decide(0.0, 1.5, [walking(3.3, 1.2, sigma=0.36, age=0.5)], [], P, side_floor_mps=FLOOR)       # the sliver of a standing box in the closed-loop scene
    assert d.comfort_cap == pytest.approx(FLOOR)


def test_the_relative_floor_can_be_switched_off():
    d = decide(0.0, 1.5, [walking(3.3, 1.4)], [], P, cfg=DecisionConfig(side_floor_relative=False), side_floor_mps=FLOOR)
    assert d.comfort_cap == pytest.approx(FLOOR)


def test_a_precaution_asks_for_no_braking_once_the_car_is_at_its_conflict_point():
    d = decide(0.0, 5.0, [conflict(0.0, False)], [], P, side_floor_mps=FLOOR)
    assert d.brake_request == 0.0 and not d.emergency and d.regime == Regime.COMFORT and d.comfort_cap == pytest.approx(FLOOR)
    assert decide(0.0, 5.0, [conflict(0.0, True)], [], P, side_floor_mps=FLOOR).emergency       # a real obstacle at the bumper is still an emergency


def test_the_cause_carries_the_floor_the_velocity_and_the_age_it_was_based_on():
    d = decide(0.0, 1.5, [walking(3.3, 1.4, sigma=0.1, age=12.0)], [], P, side_floor_mps=FLOOR)
    c = d.comfort_cause.to_dict()
    assert c["floor_mps"] == pytest.approx(FLOOR + lead_speed_low(1.4, 0.1, P)) and c["v_along_mps"] == 1.4 and c["v_sigma_mps"] == 0.1 and c["age_s"] == 12.0


def test_find_conflicts_passes_the_age_of_the_track_on():
    (c,) = conflicts(track(30.0, 3.5))
    assert c.age_s == 5.0


def test_the_slow_down_to_the_floor_is_judged_on_the_nominal_distance():
    d = decide(0.0, 2.6, [conflict(1.6, False)], [], P, side_floor_mps=FLOOR)               # a little above the floor, inside margin plus covariance
    assert d.brake_request == 0.0 and d.regime == Regime.COMFORT and not d.emergency        # (d_eff is negative here: the old rule braked at the maximum)
    assert d.comfort_cap == pytest.approx(FLOOR)                                            # the caps still carry the margin


def test_a_fast_approach_to_a_precaution_still_brakes_in_time():
    d = decide(0.0, 10.0, [conflict(12.0, False)], [], P, side_floor_mps=FLOOR)
    expected = (10.0 ** 2 - FLOOR ** 2) / (2.0 * (12.0 - 10.0 * P.delay_s))
    assert d.regime == Regime.PROGRESSIVE and d.brake_request == pytest.approx(expected)


def virtual_hazard(d, floor):
    c = replace(conflict(d, True), track_id=-2, state="virtual", sigma_d_m=0.0, margin_weight=0.0)
    return VirtualHazard("side_right", d + 3.5, 0.8, 0.4, d, c, floor)


def test_a_hidden_strip_beside_the_car_is_not_an_emergency_when_the_speed_is_a_hair_above_the_floor():
    d = decide(0.0, FLOOR + 0.04, [], [virtual_hazard(0.0, FLOOR)], P)             # the car creeps beside a parked car: this used to brake at the maximum, again and again
    assert d.brake_request == 0.0 and not d.emergency and d.regime == Regime.COMFORT and d.comfort_cap == pytest.approx(FLOOR)


def test_the_passing_speed_is_the_floor_of_a_precaution_and_of_a_hidden_strip():
    cfg = DecisionConfig(passing_speed_mps=PASS)
    for d in (decide(0.0, 13.9, [conflict(30.0, False)], [], P, cfg=cfg), decide(0.0, 13.9, [], [virtual_hazard(30.0, FLOOR)], P, cfg=cfg)):
        assert d.comfort_cap >= PASS and d.brake_request == 0.0 and d.regime == Regime.COMFORT         # 50 km/h slows to 40, gently


def test_a_hidden_corridor_ahead_and_an_object_in_the_lane_keep_their_stop_distance():
    cfg = DecisionConfig(passing_speed_mps=PASS)
    blind = decide(0.0, 13.9, [], [virtual_hazard(10.0, 0.0)], P, cfg=cfg)
    assert blind.comfort_cap < PASS
    lane = decide(0.0, 13.9, [conflict(10.0, True)], [], P, cfg=cfg)
    assert lane.comfort_cap < PASS and lane.brake_request > 0.0


def test_without_a_passing_speed_the_floor_is_the_one_that_was_passed_in():
    d = decide(0.0, 1.5, [conflict(3.3, False)], [], P, side_floor_mps=FLOOR)
    assert d.comfort_cap == pytest.approx(FLOOR)