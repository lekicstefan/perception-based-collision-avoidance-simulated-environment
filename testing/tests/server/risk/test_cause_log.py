"""Cause log and reaction events (step 7.12)."""
import json

import pytest

from server.risk.cause_log import CauseLog, read_log, reaction_events
from server.risk.decision import decide
from server.risk.risk_level import BUNDLES, RiskLevel
from testing.tests.server.risk.test_decision import conflict, virtual

P = BUNDLES[RiskLevel.MEDIUM].apply()


def test_a_record_has_the_values_and_the_three_causes():
    log = CauseLog()
    rec = log.record(decide(1.5, 10.0, [conflict(7)], [], P, risk_level=2), command_id=41)
    assert rec["t"] == 1.5 and rec["command_id"] == 41 and rec["speed"] == 10.0 and rec["risk_level"] == 2
    assert rec["comfort_cause"]["label"] == "track:7" and rec["safety_cause"]["label"] == "track:7" and rec["brake_cause"]["kind"] == "none"
    assert rec["comfort_cap"] == rec["raw_comfort_cap"] and not rec["comfort_held"] and log.records == [rec]


def test_the_output_values_replace_the_raw_ones_in_the_record():
    log = CauseLog()
    d = decide(0.0, 10.0, [conflict(7)], [], P)
    rec = log.record(d, comfort_cap=9.0, safety_cap=12.0, brake_request=1.0, emergency=False)
    assert (rec["comfort_cap"], rec["safety_cap"], rec["brake_request"]) == (9.0, 12.0, 1.0)
    assert rec["raw_comfort_cap"] == pytest.approx(d.comfort_cap)


def test_the_file_is_one_json_object_per_line_and_infinity_is_null(tmp_path):
    path = tmp_path / "causes.jsonl"
    log = CauseLog(path)
    a = log.record(decide(0.0, 10.0, [], [], P))
    b = log.record(decide(0.1, 10.0, [conflict(7)], [virtual(20.0)], P))
    log.close()
    lines = path.read_text().splitlines()
    assert len(lines) == 2 and read_log(path) == [json.loads(json.dumps(a)), json.loads(json.dumps(b))]
    assert a["comfort_cause"]["d_path_m"] is None and b["n_virtual"] == 1


def test_a_cap_held_by_the_smoothing_keeps_the_cause_that_set_it():
    log = CauseLog()
    log.record(decide(0.0, 10.0, [conflict(1)], [], P))
    held = log.record(decide(0.1, 10.0, [], [], P), comfort_cap=12.0, comfort_held=True)
    assert held["comfort_cause"]["label"] == "track:1" and held["comfort_held"] and held["safety_cause"]["kind"] == "none"
    released = log.record(decide(0.2, 10.0, [], [], P), comfort_cap=14.0)
    assert released["comfort_cause"]["kind"] == "none"


def test_a_held_cap_with_no_earlier_cause_logs_the_present_one():
    log = CauseLog()
    rec = log.record(decide(0.0, 10.0, [conflict(1)], [], P), comfort_held=True)
    assert rec["comfort_cause"]["label"] == "track:1"


def test_a_latched_emergency_keeps_the_cause_that_started_it():
    log = CauseLog()
    log.record(decide(0.0, 20.0, [conflict(5, d=8.0)], [], P))
    rec = log.record(decide(0.1, 20.0, [], [], P), brake_request=8.0, emergency=True, latched=True)
    assert rec["brake_cause"]["label"] == "track:5" and rec["latched"] and rec["emergency"]


def test_without_a_latch_no_hazard_means_no_brake_cause():
    log = CauseLog()
    log.record(decide(0.0, 20.0, [conflict(5, d=8.0)], [], P))
    assert log.record(decide(0.1, 20.0, [], [], P))["brake_cause"]["kind"] == "none"


def cause(label, kind="track", source=1):
    return {"label": label, "kind": kind, "source_id": source}


def rec(t, speed, cap, brake=0.0, emergency=False, label="track:1", regime=1):
    return {"t": t, "speed": speed, "comfort_cap": cap, "brake_request": brake, "emergency": emergency, "regime": regime,
            "comfort_cause": cause(label), "brake_cause": cause(label)}


def test_a_cap_event_is_a_stretch_with_the_comfort_cap_below_the_speed():
    recs = [rec(i * 0.1, 10.0, 6.0) for i in range(6)] + [rec(0.6, 10.0, 12.0)] + [rec(2.0 + i * 0.1, 10.0, 5.0) for i in range(3)]
    ev = reaction_events(recs)
    assert [(e["type"], e["label"], e["n"]) for e in ev] == [("cap", "track:1", 6), ("cap", "track:1", 3)]      # the gap of 1.4 s splits it
    assert ev[0]["t_start"] == 0.0 and ev[0]["t_end"] == pytest.approx(0.5) and ev[0]["min_cap"] == 6.0 and ev[1]["min_cap"] == 5.0


def test_a_brake_event_has_its_peak_and_regime():
    recs = [rec(0.0, 10.0, 12.0), rec(0.1, 10.0, 12.0, brake=3.0, regime=2), rec(0.2, 10.0, 12.0, brake=5.0, regime=3), rec(0.3, 10.0, 12.0, brake=2.0, regime=2)]
    (e,) = reaction_events(recs)
    assert (e["type"], e["n"], e["peak_brake"], e["regime_max"], e["t_start"]) == ("brake", 3, 5.0, 3, 0.1)


def test_an_emergency_without_a_request_value_is_a_brake_event():
    (e,) = reaction_events([rec(0.0, 10.0, 12.0, brake=0.0, emergency=True, regime=4)])
    assert e["type"] == "brake" and e["regime_max"] == 4


def test_events_of_different_causes_are_separate():
    recs = [rec(0.0, 10.0, 6.0, label="track:1"), rec(0.1, 10.0, 6.0, label="occlusion:side_left@28.5m"), rec(0.2, 10.0, 6.0, label="track:1")]
    ev = reaction_events(recs)
    assert sorted((e["label"], e["n"]) for e in ev) == [("occlusion:side_left@28.5m", 1), ("track:1", 2)]


def test_a_speed_below_the_cap_is_no_reaction():
    assert reaction_events([rec(i * 0.1, 5.0, 6.0) for i in range(5)]) == []


def test_the_whole_chain_from_the_decision_to_the_events():
    log = CauseLog()
    for i in range(3):
        log.record(decide(i * 0.1, 20.0, [conflict(7, d=8.0)], [], P))
    ev = reaction_events(log.records)
    assert sorted((e["type"], e["label"], e["n"]) for e in ev) == [("brake", "track:7", 3), ("cap", "track:7", 3)]
    assert all(e["kind"] == "track" and e["source_id"] == 7 for e in ev)