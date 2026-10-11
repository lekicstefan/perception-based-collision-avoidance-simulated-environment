"""Raw two-frame differencing baseline (step 6.8)."""
import numpy as np
import pytest

from server.tracking.raw_baseline import raw_static, raw_velocity
from server.tracking.kalman import ConstantVelocityFilter
from server.tracking.tracker import Tracker
from testing.tests.server.tracking.test_tracker import DT, wc


def test_the_raw_velocity_is_the_difference_over_the_time():
    assert raw_velocity([10.0, 0.0], 1.0, [10.5, -0.2], 1.1) == pytest.approx([5.0, -2.0])
    assert raw_velocity([1.0, 1.0], 1.0, [2.0, 2.0], 1.0) is None and raw_velocity([1.0, 1.0], 1.0, [2.0, 2.0], 0.9) is None


def test_the_naive_static_flag_is_a_speed_threshold():
    assert raw_static([0.3, 0.0]) and not raw_static([0.4, 0.4])


def test_a_track_has_a_raw_velocity_from_its_second_measurement_on():
    tr = Tracker()
    tr.step([wc(20.0, 0.0, 0.0)], 0.0)
    assert tr.tracks[0].raw_velocity is None
    tr.step([wc(20.5, 0.0, DT)], DT)
    assert tr.tracks[0].raw_velocity == pytest.approx([5.0, 0.0])


def test_noise_turns_into_a_large_raw_velocity_error_that_the_filter_does_not_have():
    rng = np.random.default_rng(5)
    r = np.eye(2) * 0.3 ** 2
    z = np.array([20.0, 0.0]) + rng.normal(0.0, 0.3, 2)
    f = ConstantVelocityFilter.initialise(z, r, 0.0)                       # a still object measured with 0.3 m noise
    prev, raw, filt = z, [], []
    for k in range(1, 80):
        z = np.array([20.0, 0.0]) + rng.normal(0.0, 0.3, 2)
        f.update(z, r, k * DT)
        v = raw_velocity(prev, (k - 1) * DT, z, k * DT)
        prev = z
        if k >= 20:
            raw.append(np.hypot(*v))
            filt.append(f.speed)
    assert np.sqrt(np.mean(np.square(raw))) > 3.0 * np.sqrt(np.mean(np.square(filt)))