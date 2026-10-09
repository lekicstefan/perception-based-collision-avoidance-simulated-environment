"""Tests for the cluster geometry (step 5.5), on synthetic scenes (testing/support/synthetic_scene.py)."""
import math

import numpy as np
import pytest

from server.perception.cluster_geometry import GeometryExtractor, world_xy
from server.perception.ground import GroundRemover
from server.perception.segmentation import Segmenter
from testing.support.synthetic_scene import Scene, make_lidar

LIDAR = make_lidar()


def geometries(scene, size_prior_m=1.0, **render):
    img, truth, geo = scene.render(LIDAR, **render)
    seg = Segmenter(geo).segment(img, GroundRemover(geo).candidates(img))
    return GeometryExtractor(geo, size_prior_m).extract_all(img, seg)


def largest(gs):
    return max(gs, key=lambda g: g.n_cells)


def test_a_car_seen_from_behind():
    gs = geometries(Scene().add_box(20, 0, 4.5, 1.8, 1.5))              # rear face at x = 17.75, 1.8 m wide
    assert len(gs) == 1
    g = gs[0]
    assert g.nearest_point[0] == pytest.approx(17.75, abs=0.05)
    assert g.nearest_range_m == pytest.approx(math.hypot(17.75 - 1.3, 1.8 - g.nearest_point[2]), abs=0.05)
    assert g.box_xy[0] == pytest.approx(17.75, abs=0.05) and g.box_xy[2] == pytest.approx(-0.9, abs=0.15)
    assert g.box_xy[3] == pytest.approx(0.9, abs=0.15)
    assert g.width_m == pytest.approx(1.8, abs=0.2)
    assert g.top_m == pytest.approx(1.5, abs=0.05)                      # the roof
    assert 1.0 < g.height_m <= 1.5                                      # the lowest rim is removed with the ground
    assert g.centre_xy[1] == pytest.approx(0.0, abs=0.1)
    assert g.points.shape == (g.n_cells, 3)
    assert g.bearing == pytest.approx(0.0, abs=0.02)


def test_the_corrected_centre_lies_behind_the_nearest_surface_by_half_the_depth_prior():
    g = geometries(Scene().add_box(30, 0, 0.6, 0.5, 1.7), size_prior_m=1.0)[0]       # a person, only the front is visible
    assert g.depth_m < 0.1
    assert g.centre_xy[0] - g.nearest_point[0] == pytest.approx(0.5, abs=0.05)
    g2 = geometries(Scene().add_box(30, 0, 0.6, 0.5, 1.7), size_prior_m=2.0)[0]
    assert g2.centre_xy[0] - g2.nearest_point[0] == pytest.approx(1.0, abs=0.05)
    assert g.width_m == pytest.approx(0.5, abs=0.2) and g.top_m == pytest.approx(1.55, abs=0.2)


def test_a_visible_depth_larger_than_the_prior_is_used():
    g = geometries(Scene().add_box(20, 0, 4.5, 1.8, 1.5), size_prior_m=0.2)[0]       # the roof is visible for 0.7 m
    assert g.depth_m > 0.5
    assert g.centre_xy[0] - g.nearest_point[0] == pytest.approx(g.depth_m / 2, abs=0.05)


@pytest.mark.parametrize("y", [-9, -6, 0, 6, 9])
def test_the_corrected_centre_beats_the_centroid_from_every_side(y):
    g = largest(geometries(Scene().add_box(25, y, 4.5, 1.8, 1.5)))
    true = np.array([25.0, y])
    assert np.linalg.norm(g.centre_xy - true) < np.linalg.norm(g.centroid_xy - true)
    assert g.nearest_point[0] == pytest.approx(22.75, abs=0.1)          # the nearest point stays on the rear face


def test_world_position_follows_the_pose():
    assert world_xy((10.0, 0.0), (5.0, 7.0, 0.0, 0.0, 0.0, 0.0)) == pytest.approx([15.0, 7.0])
    assert world_xy((10.0, 0.0), (5.0, 7.0, 0.0, math.pi / 2, 0.0, 0.0)) == pytest.approx([5.0, 17.0])    # facing left
    assert world_xy((0.0, 2.0), (0.0, 0.0, 0.0, math.pi / 2, 0.0, 0.0)) == pytest.approx([-2.0, 0.0])


def test_heights_are_measured_from_the_ground_plane_of_the_pitched_car():
    g = largest(geometries(Scene().add_box(20, 0, 4.5, 1.8, 1.5), pitch=-math.radians(3)))
    assert g.top_m == pytest.approx(1.5, abs=0.15)