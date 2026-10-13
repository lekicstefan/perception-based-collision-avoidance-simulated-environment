"""Ego path and corridor (step 7.1)."""
import math

import numpy as np
import pytest

from server.risk.ego_path import (Corridor, EgoGeometry, EgoPathConfig, EgoState, curvature, make_corridor, path_coordinates,
                                  point_on_path)

G = EgoGeometry(wheelbase_m=2.7, length_m=4.4, width_m=1.8, rear_overhang_m=0.9)


def test_the_front_of_the_car_is_ahead_of_the_rear_axle():
    assert G.front_offset_m == pytest.approx(3.5)


def test_curvature_from_the_steering_angle():
    k = curvature(EgoState(10.0, steering_rad=0.1), G)
    assert k == pytest.approx(math.tan(0.1) / 2.7) and k > 0                                  # left
    assert curvature(EgoState(10.0, steering_rad=-0.1), G) == pytest.approx(-k)
    assert curvature(EgoState(10.0), G) == 0.0


def test_curvature_from_the_yaw_rate_and_the_low_speed_fallback():
    cfg = EgoPathConfig(source="yaw_rate")
    assert curvature(EgoState(10.0, yaw_rate_rps=0.2), G, cfg) == pytest.approx(0.02)
    assert curvature(EgoState(0.2, yaw_rate_rps=0.5, steering_rad=0.1), G, cfg) == pytest.approx(math.tan(0.1) / 2.7)   # too slow for omega / v


def test_the_curvature_is_limited_and_an_unknown_source_is_an_error():
    assert curvature(EgoState(10.0, steering_rad=1.2), G) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        curvature(EgoState(10.0), G, EgoPathConfig(source="magic"))


def test_straight_path_coordinates_are_x_and_y():
    s, lat = path_coordinates(np.array([[10.0, 1.0], [-2.0, -0.5]]), 0.0)
    assert s == pytest.approx([10.0, -2.0]) and lat == pytest.approx([1.0, -0.5])


@pytest.mark.parametrize("k", [0.05, -0.05])
def test_points_on_the_arc_have_zero_lateral_offset_and_their_arc_length(k):
    s_true = np.array([0.0, 5.0, 20.0, 40.0])
    s, lat = path_coordinates(point_on_path(s_true, k), k)
    assert s == pytest.approx(s_true) and lat == pytest.approx(0.0, abs=1e-9)


def test_the_lateral_offset_is_positive_to_the_left_on_both_turns():
    for k in (0.05, -0.05):
        p = point_on_path([15.0], k)[0]
        heading = 15.0 * k
        left = np.array([-math.sin(heading), math.cos(heading)])                              # the left normal of the arc at that point
        _, lat = path_coordinates(p + 1.0 * left, k)
        assert lat == pytest.approx(1.0, abs=1e-6)
        _, lat = path_coordinates(p - 0.5 * left, k)
        assert lat == pytest.approx(-0.5, abs=1e-6)


def test_the_corridor_is_the_width_plus_the_margin():
    c = make_corridor(EgoState(10.0), G, EgoPathConfig(lateral_margin_m=0.3))
    assert c.half_width_m == pytest.approx(1.2) and c.front_offset_m == pytest.approx(3.5)
    pts = np.array([[20.0, 1.1], [20.0, 1.3], [20.0, -1.19], [150.0, 0.0], [-10.0, 0.0]])
    assert list(c.inside(pts)) == [True, False, True, False, False]
    assert bool(c.inside(np.array([[20.0, 1.3]]), extra_m=0.2)[0])                         # a point with a radius of 0.2 m


def test_the_distance_ahead_counts_from_the_front_bumper():
    c = make_corridor(EgoState(10.0), G)
    assert c.distance_ahead([23.5, 2.0]) == pytest.approx([20.0, 0.0])


def test_the_corridor_follows_a_turn():
    c = make_corridor(EgoState(10.0, steering_rad=0.1), G)
    straight = make_corridor(EgoState(10.0), G)
    p = np.array([[30.0, 0.0]])
    assert straight.inside(p)[0] and not c.inside(p)[0]                                   # the turn takes the corridor away from the x axis
    assert c.inside(point_on_path([30.0], c.k))[0]
    assert c.centreline(5).shape == (5, 2)