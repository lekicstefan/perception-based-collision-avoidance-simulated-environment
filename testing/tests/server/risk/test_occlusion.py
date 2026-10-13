"""Virtual hazards at the edge of what is visible (step 7.10), on synthetic ray-cast scenes."""
import numpy as np
import pytest

from server.perception.ground import GroundConfig, GroundRemover
from server.perception.range_image import CellState
from server.risk.ego_path import EgoGeometry, EgoState, make_corridor
from server.risk.hazards import HazardConfig
from server.risk.occlusion import OcclusionConfig, find_virtual_hazards
from server.risk.visibility import ViewField, measure_view
from testing.support.synthetic_scene import Scene, make_lidar

LIDAR = make_lidar()
G = EgoGeometry()
PARKED = lambda y=3.0: Scene().add_box(30.0, y, 4.5, 1.8, 1.5)             # a parked car beside the road: front face 27.75 m, rear end 32.25 m


def hazards(scene, v=10.0, vocc=2.0, travel=0.0, steering=0.0, edit=None, cfg=OcclusionConfig(), hcfg=HazardConfig()):
    img, _, geo = scene.render(LIDAR)
    if edit:
        edit(img)
    view = ViewField(img, GroundRemover(geo, GroundConfig()).ground_mask(img), geo)
    corridor = make_corridor(EgoState(v, steering_rad=steering), G)
    m = measure_view(view, corridor)
    return find_virtual_hazards(view, corridor, m, v, vocc, travel, G, cfg, hcfg)


def test_an_empty_road_has_no_virtual_hazard():
    assert hazards(Scene()) == []


def test_a_parked_car_hides_a_strip_and_gets_a_side_hazard_on_its_side_only():
    hs = hazards(PARKED())
    assert [h.kind for h in hs] == ["side_left"]
    h = hs[0]
    assert 27.0 < h.s_m < 40.0 and 0.0 < h.gap_m < 2.5 and h.conflict.state == "virtual" and h.conflict.track_id < 0
    assert h.tau_first_s == pytest.approx(h.gap_m / 2.0) and h.d_path_m == pytest.approx(h.s_m - 3.5)


def test_a_parked_car_on_the_right_gives_a_right_hazard():
    assert [h.kind for h in hazards(PARKED(-3.0))] == ["side_right"]


def test_a_car_far_to_the_side_hides_nothing_that_matters():
    assert hazards(PARKED(9.0)) == []                                     # its shadow is outside the strip of 3 m beside the corridor


def test_a_faster_pedestrian_is_at_the_edge_sooner():
    slow, fast = hazards(PARKED(), vocc=1.5)[0], hazards(PARKED(), vocc=2.5)[0]
    assert fast.tau_first_s < slow.tau_first_s and fast.s_m == pytest.approx(slow.s_m)


def test_a_car_that_is_past_before_anything_can_be_there_has_no_conflict():
    near = Scene().add_box(8.0, 3.0, 2.0, 1.8, 1.5)                          # a parked car whose shadow starts a few metres ahead
    slow = hazards(near, v=5.0, hcfg=HazardConfig(time_margin_s=0.0))
    fast = hazards(near, v=20.0, hcfg=HazardConfig(time_margin_s=0.0))
    assert [h.kind for h in slow] == ["side_left"] and fast == []              # at 20 m/s the rear is past the station in 0.4 s, the pedestrian needs 0.55 s


def test_a_standing_car_never_clears():
    hs = hazards(PARKED(), v=0.0)
    assert hs and hs[0].conflict.t_clear_s == float("inf")


def test_the_distance_counts_from_the_bumper_at_the_time_of_effect():
    d0 = hazards(PARKED(), travel=0.0)[0].d_path_m
    d1 = hazards(PARKED(), travel=0.8)[0].d_path_m
    assert d0 - d1 == pytest.approx(0.8)


def test_a_crest_gives_a_stationary_hazard_at_the_edge_of_the_visible_road():
    hs = hazards(Scene([(0.0, 0.0), (20.0, 0.0875), (30.0, -0.0875)]))
    blind = [h for h in hs if h.kind == "blind_corridor"]
    assert len(blind) == 1 and 15.0 < blind[0].d_path_m < 40.0 and blind[0].tau_first_s == 0.0 and blind[0].gap_m == 0.0


def test_a_blind_stretch_from_missing_returns_is_a_hazard_too():
    def edit(img):
        img.state[:, 90:151] = CellState.NO_RETURN
        img.range_m[:, 90:151] = np.nan
    assert any(h.kind == "blind_corridor" for h in hazards(Scene(), edit=edit))


def test_a_real_hazard_in_the_corridor_is_not_duplicated_as_blind():
    assert all(h.kind != "blind_corridor" for h in hazards(Scene().add_box(30.0, 0.0, 4.5, 1.8, 1.5)))


def test_it_can_be_switched_off():
    assert hazards(PARKED(), cfg=OcclusionConfig(enabled=False)) == []


def test_the_nearest_comes_first_and_the_cause_names_the_place():
    hs = hazards(Scene().add_box(30.0, 3.0, 4.5, 1.8, 1.5).add_box(30.0, -3.0, 4.5, 1.8, 1.5))
    assert [h.d_path_m for h in hs] == sorted(h.d_path_m for h in hs) and len(hs) == 2
    assert hs[0].cause.startswith("occlusion:side_") and "m" in hs[0].cause


def test_a_virtual_hazard_has_no_margin_and_no_distance_error_and_a_floor_for_the_side_ones():
    hs = hazards(PARKED(), vocc=2.5)
    assert hs[0].conflict.margin_weight == 0.0 and hs[0].conflict.sigma_d_m == 0.0 and hs[0].floor_speed_mps == 2.5
    crest = hazards(Scene([(0.0, 0.0), (20.0, 0.0875), (30.0, -0.0875)]))
    assert [h.floor_speed_mps for h in crest if h.kind == "blind_corridor"] == [0.0]