"""Delay accounting and the uncertainty of the distance (steps 7.2 and 7.6)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from server.risk.stopping import StoppingParams, effective_distance, lead_speed_low
from server.risk.timing import DelayAccounting, DelayConfig, ego_travel, predict_track
from server.risk.uncertainty import distance_sigma, path_tangent, speed_along_path
from server.tracking.kalman import process_noise

POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def out(position=(50.0, 0.0), velocity=(0.0, 0.0), t=1.0, cov=None, static=False):
    cov = np.diag([0.25, 0.25, 0.04, 0.04]) if cov is None else cov
    return SimpleNamespace(id=7, t=t, position=np.array(position, float), velocity=np.array(velocity, float), cov=cov, static=static,
                           velocity_known=True, state="confirmed", existence=1.0, age_s=3.0)


def test_the_horizon_is_the_smoothed_latency_plus_the_transit_time():
    d = DelayAccounting(DelayConfig(initial_latency_s=0.05, transit_s=0.005))
    assert d.horizon_s == pytest.approx(0.055) and d.time_of_effect(10.0) == pytest.approx(10.055)


def test_the_smoothed_latency_follows_the_samples_and_a_stall_is_limited():
    d = DelayAccounting(DelayConfig(initial_latency_s=0.05, smoothing=0.5, max_latency_s=0.5))
    d.update(0.10)
    assert d.latency_s == pytest.approx(0.075)
    for _ in range(30):
        d.update(0.10)
    assert d.latency_s == pytest.approx(0.10, abs=1e-3)
    d.update(5.0)                                                                  # a stalled cycle counts as at most max_latency_s
    assert d.latency_s == pytest.approx(0.10 + 0.5 * (0.5 - 0.10))


def test_a_static_track_stays_and_its_covariance_grows_with_the_interval():
    p = predict_track(out(), t_effect=1.2, pose_data=POSE, sigma_a=1.4)
    assert p.position == pytest.approx([50.0, 0.0]) and p.t == pytest.approx(1.2)
    q = process_noise(0.2, 1.4)
    assert p.cov[0, 0] == pytest.approx(0.25 + 0.04 * 0.2 ** 2 + q[0, 0]) and p.cov[2, 2] == pytest.approx(0.04 + q[2, 2])


def test_a_moving_track_is_moved_with_its_velocity():
    p = predict_track(out(velocity=(-5.0, 1.0)), t_effect=1.2, pose_data=POSE, sigma_a=1.4)
    assert p.position == pytest.approx([49.0, 0.2]) and p.velocity == pytest.approx([-5.0, 1.0])


def test_no_time_no_change_and_never_backwards():
    p = predict_track(out(velocity=(-5.0, 0.0)), t_effect=1.0, pose_data=POSE, sigma_a=1.4)
    assert p.position == pytest.approx([50.0, 0.0]) and p.cov == pytest.approx(np.diag([0.25, 0.25, 0.04, 0.04]))
    earlier = predict_track(out(velocity=(-5.0, 0.0)), t_effect=0.5, pose_data=POSE, sigma_a=1.4)
    assert earlier.position == pytest.approx([50.0, 0.0])


def test_the_result_is_in_the_vehicle_frame_of_the_data_time():
    pose = (10.0, 0.0, 0.0, math.pi / 2, 0.0, 0.0)                                  # the car at (10, 0) looking along +y
    p = predict_track(out(position=(10.0, 30.0), velocity=(0.0, -4.0), t=1.0, cov=np.diag([4.0, 1.0, 0.5, 0.1])), t_effect=2.0,
                      pose_data=pose, sigma_a=0.0)
    assert p.position == pytest.approx([26.0, 0.0], abs=1e-9)                       # 30 m ahead, 4 m closer after one second
    assert p.velocity == pytest.approx([-4.0, 0.0], abs=1e-9)                       # coming towards the car
    assert p.pos_cov[0, 0] == pytest.approx(1.0 + 0.1) and p.pos_cov[1, 1] == pytest.approx(4.0 + 0.5)     # the axes swap; velocity term dt^2 * var


def test_the_car_drives_on_at_its_speed():
    assert ego_travel(15.0, 0.055) == pytest.approx(0.825) and ego_travel(-1.0, 0.1) == 0.0 and ego_travel(10.0, -0.1) == 0.0


def test_the_distance_sigma_is_the_covariance_projected_on_the_path():
    c = np.diag([4.0, 1.0])
    assert distance_sigma(c, [1.0, 0.0]) == pytest.approx(2.0) and distance_sigma(c, [0.0, 1.0]) == pytest.approx(1.0)
    assert distance_sigma(c, [math.sqrt(0.5), math.sqrt(0.5)]) == pytest.approx(math.sqrt(2.5))
    assert distance_sigma(c, [1.0, 0.0], ego_sigma_m=1.5) == pytest.approx(math.sqrt(4.0 + 2.25))          # the car's own pose error adds


def test_the_path_tangent_turns_with_the_arc():
    assert path_tangent(10.0, 0.0) == pytest.approx([1.0, 0.0])
    t = path_tangent(10.0, 0.05)
    assert t == pytest.approx([math.cos(0.5), math.sin(0.5)]) and np.hypot(*t) == pytest.approx(1.0)


def test_the_speed_along_the_path_and_its_uncertainty():
    v, s = speed_along_path([6.0, 3.0], np.diag([0.25, 4.0]), [1.0, 0.0])
    assert v == pytest.approx(6.0) and s == pytest.approx(0.5)
    v, s = speed_along_path([-2.0, 0.0], np.diag([0.25, 4.0]), [1.0, 0.0])
    assert v == pytest.approx(-2.0) and lead_speed_low(v, s, StoppingParams()) == 0.0                         # coming towards the car: no allowance


def test_a_larger_uncertainty_gives_a_shorter_effective_distance():
    p = StoppingParams(epsilon=0.02, kappa=1.5)
    assert effective_distance(30.0, p, sigma_d=distance_sigma(np.diag([1.0, 1.0]), [1.0, 0.0])) < effective_distance(
        30.0, p, sigma_d=distance_sigma(np.diag([0.25, 0.25]), [1.0, 0.0]))