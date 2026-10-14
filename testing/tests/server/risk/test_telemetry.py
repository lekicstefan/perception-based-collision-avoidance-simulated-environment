"""The telemetry message of a scan (step 7.14)."""
import json
import math

import numpy as np
import pytest

from server.communication import protocol as P
from server.risk.telemetry import SCHEMA, build_telemetry, clean
from server.perception.ground import GroundConfig, GroundRemover
from server.risk.ego_path import EgoGeometry, EgoState
from server.risk.stage import RiskStage, StageConfig
from server.tracking.outputs import track_output
from server.tracking.tracker import Tracker
from testing.support.synthetic_scene import Scene, make_lidar
from testing.tests.server.tracking.test_extent import seen
from testing.tests.server.tracking.test_tracker import DT

LIDAR = make_lidar()
EGO = EgoState(10.0)


def scan(scene, t):
    img, _, geo = scene.render(LIDAR)
    img = type(img)(t, img.seq, img.pose, 10.0, 0.0, 0.0, img.range_m, img.state)
    return img, GroundRemover(geo, GroundConfig()).ground_mask(img), geo


def tracks_of(x_near):
    tr = Tracker()
    for k in range(6):
        tr.step([seen([x_near, -0.9], [x_near + 4.5, 0.9], k * DT)], k * DT)
    return [track_output(t) for t in tr.tracks]


def stage(geo, **kw):
    return RiskStage(EgoGeometry(), geo, StageConfig(**kw))


def run(scene, outputs, t=0.5, **kw):
    img, ground, geo = scan(scene, t)
    return stage(geo, **kw).step(img, ground, EGO, outputs)


def test_clean_makes_everything_json_safe():
    x = clean({"a": np.float64(1.5), "b": math.inf, "c": (np.int64(2), math.nan), "d": np.array([1.0, np.inf]), "e": np.bool_(True), 3: None})
    assert x == {"a": 1.5, "b": None, "c": [2, None], "d": [1.0, None], "e": True, "3": None}
    json.dumps(x, allow_nan=False)


def test_an_empty_road_message_has_the_fields_and_no_tracks():
    msg = build_telemetry(run(Scene(), []))
    json.dumps(msg, allow_nan=False)                                                  # no NaN, no inf
    assert msg["schema"] == SCHEMA == 1 and msg["t"] == 0.5 and msg["command_id"] == 1 and msg["speed"] == 10.0
    assert msg["tracks"] == [] and msg["virtual"] == []
    assert msg["command"]["safety_cap"] == 40.0 and msg["command"]["risk_level"] == 0 and msg["risk"]["level"] == 0
    assert msg["decision"]["comfort_cause"]["kind"] == "none"
    assert msg["collision"] is None and msg["timing"]["stage_ms"]["total"] > 0


def test_a_car_ahead_appears_as_a_track_with_its_box():
    msg = build_telemetry(run(Scene().add_box(30.0, 0.0, 4.5, 1.8, 1.5), tracks_of(27.75)))
    json.dumps(msg, allow_nan=False)
    assert len(msg["tracks"]) == 1
    tr = msg["tracks"][0]
    assert tr["id"] == 1 and tr["state"] == "confirmed" and 27.0 < tr["x"] < 33.0 and abs(tr["y"]) < 1.0
    assert len(tr["corners"]) == 4 and all(len(c) == 2 for c in tr["corners"])
    assert msg["collision"]["method"] in ("sampled", "deterministic", "mixed") and msg["decision"]["n_hazards"] >= 1
    assert msg["command"]["comfort_cap"] <= msg["command"]["safety_cap"] < 40.0
    assert msg["raw"]["safety_cap"] == pytest.approx(msg["command"]["safety_cap"])        # the first cycle: nothing is held yet


def test_the_message_fits_the_wire_and_comes_back():
    msg = build_telemetry(run(Scene().add_box(30.0, 0.0, 4.5, 1.8, 1.5), tracks_of(27.75)))
    kind, back = P.parse_command_message(P.build_telemetry(msg))
    assert kind == P.CmdType.TELEMETRY and back == msg