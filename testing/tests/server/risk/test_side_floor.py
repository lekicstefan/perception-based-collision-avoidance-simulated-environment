"""A conflict that exists only because an object could reach the corridor is answered with a floor speed (step 7.15 follow-up):
a car parked beside the road is passed, not stopped for. An object whose nominal motion enters the corridor is not affected."""
import numpy as np
import pytest

from server.risk.decision import DecisionConfig, decide
from server.risk.ego_path import EgoGeometry, EgoState, make_corridor
from server.risk.hazards import Conflict, find_conflicts
from server.risk.risk_level import BUNDLES, RiskLevel
from server.risk.timing import PredictedTrack
from server.risk.stopping import Regime

P = BUNDLES[RiskLevel.MEDIUM].apply()
FLOOR = BUNDLES[RiskLevel.MEDIUM].occluded_speed_mps                   # 2.0
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