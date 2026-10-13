"""The view field and the two measurements of the risk level (step 7.8), on synthetic ray-cast scenes."""
import numpy as np
import pytest

from server.perception.ground import GroundConfig, GroundRemover
from server.perception.range_image import CellState
from server.risk.ego_path import EgoGeometry, EgoState, make_corridor
from server.risk.visibility import PointState, ViewConfig, ViewField, corridor_cells, measure_view
from testing.support.synthetic_scene import Scene, make_lidar

LIDAR = make_lidar()
G = EgoGeometry()
FRONT = G.front_offset_m                                           # 3.5 m


def view(scene, edit=None, **render):
    img, _, geo = scene.render(LIDAR, **render)
    if edit:
        edit(img)
    return ViewField(img, GroundRemover(geo, GroundConfig()).ground_mask(img), geo)


def measure(scene, speed=10.0, steering=0.0, edit=None, **render):
    return measure_view(view(scene, edit, **render), make_corridor(EgoState(speed, steering_rad=steering), G))


def test_an_empty_road_is_seen_to_the_evaluation_range():
    m = measure(Scene())
    assert m.sight_distance_m == pytest.approx(60.0) and m.limit == "clear" and m.unknown_fraction == 0.0 and not m.hazard_ahead


def test_a_car_in_the_corridor_ends_the_sight_at_its_front():
    m = measure(Scene().add_box(30.0, 0.0, 4.5, 1.8, 1.5))              # front face at 27.75 m
    assert m.limit == "hazard" and m.hazard_ahead and m.sight_distance_m == pytest.approx(27.75 - FRONT, abs=1.0)


def test_a_narrow_pedestrian_in_the_corridor_ends_it_too():
    m = measure(Scene().add_box(25.0, 0.3, 0.5, 0.5, 1.7))
    assert m.limit == "hazard" and m.sight_distance_m == pytest.approx(24.75 - FRONT, abs=1.0)


def test_a_car_beside_the_corridor_does_not():
    m = measure(Scene().add_box(30.0, 6.0, 4.5, 1.8, 1.5))
    assert m.limit == "clear" and m.sight_distance_m == pytest.approx(60.0)


def test_what_is_behind_a_hazard_is_not_unknown_cells():
    m = measure(Scene().add_box(30.0, 0.0, 0.6, 60.0, 3.0))             # a wall across the road
    assert m.limit == "hazard" and m.unknown_fraction == 0.0


def test_a_crest_hides_the_road_behind_it():
    m = measure(Scene([(0.0, 0.0), (20.0, 0.0875), (30.0, -0.0875)]))
    assert m.limit == "occluded" and 15.0 < m.sight_distance_m < 40.0 and m.unknown_fraction > 0.3


def test_a_constant_slope_and_a_hill_ahead_are_not_a_crest():
    for scene in (Scene([(0.0, 0.07)]), Scene([(0.0, -0.05)]), Scene([(0.0, 0.0), (20.0, 0.14)])):
        assert measure(scene).limit == "clear"


def test_a_single_dropped_column_is_not_an_occlusion():
    def edit(img):
        img.state[:, 120] = CellState.NO_RETURN
        img.range_m[:, 120] = np.nan
    m = measure(Scene(), edit=edit)
    assert m.limit == "clear" and 0.0 < m.unknown_fraction < 0.2


def test_no_returns_at_all_in_the_corridor_are_unknown_and_end_the_sight():
    def edit(img):
        img.state[:, 90:151] = CellState.NO_RETURN
        img.range_m[:, 90:151] = np.nan
    m = measure(Scene(), edit=edit)
    assert m.limit == "dropout" and m.unknown_fraction > 0.9 and m.sight_distance_m < 5.0


def test_a_tight_bend_leaves_the_field_of_view():
    m = measure(Scene(), steering=0.6)
    assert m.limit in ("outside", "occluded") and m.sight_distance_m < 40.0 and m.unknown_fraction > 0.2


def test_the_states_of_single_points():
    v = view(Scene().add_box(30.0, 0.0, 4.5, 1.8, 1.5))
    pts = np.array([[15.0, 0.0], [27.75, 0.0], [34.0, 0.0], [34.0, 8.0], [10.0, 40.0], [150.0, 0.0], [3.0, 0.0], [20.0, 50.0]])
    got = [PointState(int(s)) for s in v.classify(pts)]
    assert got == [PointState.FREE, PointState.BLOCKED, PointState.OCCLUDED, PointState.FREE, PointState.OUTSIDE, PointState.RANGE,
                   PointState.NEAR, PointState.OUTSIDE]


def test_the_cells_cover_the_corridor_from_the_front_bumper():
    s, lat, pts = corridor_cells(make_corridor(EgoState(10.0), G), ViewConfig())
    assert s[0] == pytest.approx(FRONT + 0.5) and s[-1] == pytest.approx(FRONT + 59.5) and lat[0] == pytest.approx(-1.2)
    assert pts.shape == (60, len(lat), 2) and pts[0, 0] == pytest.approx([FRONT + 0.5, -1.2])


def test_the_cells_follow_the_arc_on_a_bend_up_to_half_a_circle():
    c = make_corridor(EgoState(10.0, steering_rad=0.3), G)                   # k = 0.115, radius 8.7 m
    s, lat, pts = corridor_cells(c, ViewConfig())
    assert s[-1] < 0.9 * np.pi / abs(c.k) and len(s) < 60
    mid = pts[:, len(lat) // 2]
    assert c.inside(mid).all() and np.abs(c.coordinates(mid)[1]).max() < 1e-9
    assert np.abs(c.coordinates(pts.reshape(-1, 2))[1]).max() == pytest.approx(c.half_width_m)


def test_the_measurement_is_fast():
    import time
    t = time.perf_counter()
    measure(Scene().add_box(30.0, 0.0, 4.5, 1.8, 1.5))
    assert time.perf_counter() - t < 0.2                                  # the whole call including the render; only a gross slowdown fails