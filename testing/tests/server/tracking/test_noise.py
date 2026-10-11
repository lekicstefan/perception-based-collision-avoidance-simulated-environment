"""Measurement noise of a cluster (step 6.2): elongated along the line of sight, wider across it with the range."""
import math

import numpy as np
import pytest

from server.tracking.measurement import WorldCluster
from server.tracking.noise import MeasurementNoise, measurement_of


def cluster(range_m=30.0, bearing=0.0, sensor=(1.3, 0.0)):
    u = np.array([math.cos(bearing), math.sin(bearing)])
    s = np.array(sensor)
    return WorldCluster(t=2.5, pose=(0, 0, 0, 0, 0, 0), label=0, n_cells=40, nearest_xy=s + u * range_m,
                        centre_xy=s + u * (range_m + 1.0), centroid_xy=s + u * (range_m + 0.7), sensor_xy=s,
                        bearing=bearing, range_m=range_m, corners_xy=np.zeros((4, 2)), width_m=1.0, depth_m=0.5,
                        height_m=1.5, top_m=1.5)


def test_the_noise_is_small_along_the_line_of_sight_and_grows_across_it_with_the_range():
    n = MeasurementNoise()
    a10, c10 = n.sigmas(10.0)
    a50, c50 = n.sigmas(50.0)
    assert a10 == pytest.approx(a50) and a10 < c50
    assert c50 == pytest.approx(5 * c10, rel=0.05)                                      # the cell width is ~ range
    assert c50 == pytest.approx(50.0 * math.radians(0.2) / math.sqrt(12), rel=0.05)


def test_no_sigma_is_below_the_floor_and_pose_noise_adds_to_both():
    n = MeasurementNoise(floor_m=0.05)
    assert min(n.sigmas(1.0)) >= 0.05
    noisy = MeasurementNoise(pose_position_sigma_m=0.3, pose_yaw_sigma_rad=math.radians(1.0))
    assert noisy.sigmas(40.0)[0] > n.sigmas(40.0)[0] and noisy.sigmas(40.0)[1] > 3 * n.sigmas(40.0)[1]


def test_the_centre_is_less_certain_along_the_line_of_sight_than_the_nearest_point():
    n = MeasurementNoise()
    assert n.sigmas(30.0, "centre")[0] > 5 * n.sigmas(30.0, "nearest")[0]
    assert n.sigmas(30.0, "centre")[1] == pytest.approx(n.sigmas(30.0, "nearest")[1])
    with pytest.raises(ValueError):
        n.sigmas(30.0, "corner")


def test_the_covariance_is_turned_into_the_world_frame_with_the_bearing():
    n = MeasurementNoise()
    a, c = n.sigmas(40.0)
    r0 = n.covariance(40.0, 0.0)                                                         # looking along +x
    assert r0[0, 0] == pytest.approx(a ** 2) and r0[1, 1] == pytest.approx(c ** 2)
    r90 = n.covariance(40.0, math.pi / 2)                                                # looking along +y
    assert r90[1, 1] == pytest.approx(a ** 2) and r90[0, 0] == pytest.approx(c ** 2)
    r = n.covariance(40.0, 0.7)
    assert r == pytest.approx(r.T) and np.all(np.linalg.eigvalsh(r) > 0)


def test_measurement_of_a_world_cluster_gives_position_covariance_and_time():
    w = cluster(range_m=30.0, bearing=0.3)
    z, r, t = measurement_of(w, MeasurementNoise(), "nearest")
    assert z == pytest.approx(w.nearest_xy) and t == 2.5 and r.shape == (2, 2)
    zc, rc, _ = measurement_of(w, MeasurementNoise(), "centre")
    assert zc == pytest.approx(w.centre_xy) and np.trace(rc) > np.trace(r)
    with pytest.raises(ValueError):
        measurement_of(w, MeasurementNoise(), "corner")