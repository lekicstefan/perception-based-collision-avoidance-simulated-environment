"""The risk stage of a scan (step 7.13) on ray-cast scenes with tracks from the tracker."""
import pytest

from server.perception.ground import GroundConfig, GroundRemover
from server.risk.ego_path import EgoGeometry, EgoState
from server.risk.stage import RiskStage, StageConfig
from server.tracking.outputs import track_output
from server.tracking.tracker import Tracker
from testing.support.synthetic_scene import Scene, make_lidar
from testing.tests.server.tracking.test_extent import seen
from testing.tests.server.tracking.test_tracker import DT

LIDAR = make_lidar()
G = EgoGeometry()
EGO = EgoState(10.0)


def scan(scene, t, speed=10.0):
    img, _, geo = scene.render(LIDAR)
    img = type(img)(t, img.seq, img.pose, speed, 0.0, 0.0, img.range_m, img.state)
    return img, GroundRemover(geo, GroundConfig()).ground_mask(img), geo


def tracks_of(x_near):
    tr = Tracker()
    for k in range(6):
        tr.step([seen([x_near, -0.9], [x_near + 4.5, 0.9], k * DT)], k * DT)
    return [track_output(t) for t in tr.tracks]


def stage(geo, **kw):
    return RiskStage(G, geo, StageConfig(**kw))


def test_an_empty_road_gives_no_limit_and_the_lowest_risk():
    img, ground, geo = scan(Scene(), 0.5)
    r = stage(geo).step(img, ground, EGO, [])
    c = r.command
    assert c.id == 1 and c.data_timestamp == 0.5 and c.safety_cap == 40.0 and c.comfort_cap == 40.0
    assert c.brake_request == 0.0 and not c.emergency and c.risk_level == 0
    assert r.n_tracks == 0 and r.decision.n_hazards == 0


def test_a_stopped_car_ahead_limits_the_caps():
    img, ground, geo = scan(Scene().add_box(30.0, 0.0, 4.5, 1.8, 1.5), 0.5)
    r = stage(geo).step(img, ground, EGO, tracks_of(27.75))
    c = r.command
    assert r.n_tracks == 1 and r.decision.n_hazards >= 1
    assert 0.0 < c.comfort_cap <= c.safety_cap < 40.0


def test_a_car_very_close_is_an_emergency():
    img, ground, geo = scan(Scene().add_box(10.0, 0.0, 4.5, 1.8, 1.5), 0.5)
    r = stage(geo).step(img, ground, EGO, tracks_of(7.75))
    assert r.command.emergency and r.command.brake_request == 8.0 and r.command.risk_level == 2


def test_caps_come_back_only_after_the_hold():
    st = stage(scan(Scene(), 0.0)[2])
    img, ground, _ = scan(Scene().add_box(30.0, 0.0, 4.5, 1.8, 1.5), 0.5)
    low = st.step(img, ground, EGO, tracks_of(27.75)).command
    img, ground, _ = scan(Scene(), 0.6)
    r = st.step(img, ground, EGO, [])
    assert r.command.comfort_cap == pytest.approx(low.comfort_cap) and r.record["comfort_held"]       # the raw cap is 40, the output holds
    assert r.decision.comfort_cap == 40.0


def test_the_command_ids_count_up_and_the_latency_follows_the_samples():
    st = stage(scan(Scene(), 0.0)[2])
    before = st.delay.latency_s
    img, ground, _ = scan(Scene(), 0.5)
    a = st.step(img, ground, EGO, [], lead_s=0.2)
    img, ground, _ = scan(Scene(), 0.6)
    b = st.step(img, ground, EGO, [])
    assert (a.command.id, b.command.id) == (1, 2)
    assert a.latency_s > before                                                     # a sample of 0.2 s moves the average up
    assert a.t_effect == pytest.approx(0.5 + before + st.cfg.delay.transit_s)
    assert a.timings_ms["total"] > 0 and {"view", "hazards", "collision", "decision"} <= set(a.timings_ms)