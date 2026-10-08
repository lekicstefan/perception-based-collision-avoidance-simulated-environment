"""The information-boundary test (step 4.9): only allowed message types and fields exist in normal mode."""
import json
import socket
import struct
from pathlib import Path

import numpy as np
import pytest

import boundary as B
import protocol as P
from stream_log import StreamRecorder
from test_replay import write_recording

ROOT = Path(__file__).resolve().parent.parent
REAL_CALIBRATION = ROOT / "testing" / "recordings" / "dev_default_seed12345" / "calibration.json"
MINIMAL_CALIBRATION = {"schema": 1, "calibrationId": "x", "lidar": {"rows": 32, "mount": {"x": 1.0}}}


def session(mtype, obj):
    return P.build_message(mtype, 0, P.json_payload(obj))


def test_schema_matches_the_golden_copy():
    assert B.check_schema() == []


def test_a_changed_header_is_detected(monkeypatch):
    monkeypatch.setattr(P, "HEADER", struct.Struct(P.HEADER.format + "f"))     # someone adds a field
    assert any("header layout changed" in p for p in B.check_schema())


def test_oracle_messages_are_rejected_in_normal_mode():
    raw = P.build_message(P.MsgType.ORACLE, 0, P.oracle_payload(np.zeros(1, dtype=P.ORACLE_DTYPE)))
    assert any("ORACLE" in p for p in B.check_message(raw))
    assert B.check_message(raw, normal_mode=False) == []


@pytest.mark.skipif(not REAL_CALIBRATION.exists(), reason="no recorded calibration.json in the repository")
def test_the_real_calibration_respects_the_boundary():
    data = json.loads(REAL_CALIBRATION.read_text(encoding="utf-8"))
    assert B.check_message(session(P.MsgType.CALIBRATION, data)) == []


@pytest.mark.parametrize("path, value", [
    (("lidar", "fogAttenuation"), 0.1),
    (("noiseSigma",), 0.02),
    (("seed",), 12345),
    (("hazards",), [{"x": 1}]),
    (("lidar", "rangeNoiseM"), 0.03),
])
def test_forbidden_calibration_keys_are_caught(path, value):
    data = json.loads(json.dumps(MINIMAL_CALIBRATION))
    assert B.check_message(session(P.MsgType.CALIBRATION, data)) == []
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    assert B.check_message(session(P.MsgType.CALIBRATION, data)) != []


def test_session_message_keys():
    assert B.check_message(session(P.MsgType.HELLO, {"schema": 1, "run_dir": "C:/runs/x"})) == []
    assert B.check_message(session(P.MsgType.HELLO, {"schema": 1, "scenario": "fog"})) != []
    assert B.check_message(session(P.MsgType.END_OF_RUN, {"sim_time": 12.0})) == []
    assert B.check_message(session(P.MsgType.END_OF_RUN, {"sim_time": 12.0, "reason": "collision"})) != []


def test_an_open_oracle_port_is_detected():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        problems, state = B.check_ports(oracle_port=port)
        assert state["oracle"] and problems
        assert B.check_ports(normal_mode=False, oracle_port=port)[0] == []
    finally:
        listener.close()
    assert B.check_ports(oracle_port=port)[0] == []                 # closed again: fine


def test_a_normal_stream_passes(tmp_path):
    write_recording(tmp_path / "s.bin")
    rep = B.check_stream(tmp_path / "s.bin")
    assert rep.ok, rep.problems
    assert rep.counts == {"CALIBRATION": 1, "POSE": 50, "CAMERA": 34, "LIDAR": 10, "END_OF_RUN": 1}


def test_unsafe_streams_are_caught(tmp_path):
    flagged = bytearray(P.build_message(P.MsgType.POSE, 0))
    flagged[7] = 1                                                   # flags must be 0
    rec = StreamRecorder(tmp_path / "bad.bin")
    rec.write(P.build_message(P.MsgType.ORACLE, 0, P.oracle_payload(np.zeros(1, dtype=P.ORACLE_DTYPE))), 0.0)
    rec.write(bytes(flagged), 0.1)
    rec.write(b"junk", 0.2)
    rec.close()
    rep = B.check_stream(tmp_path / "bad.bin")
    assert not rep.ok and len(rep.problems) == 3


def test_recorded_runs_respect_the_boundary():
    streams = sorted((ROOT / "runs").glob("*/python/stream.bin"), key=lambda p: p.stat().st_mtime)[-3:]
    if not streams:
        pytest.skip("no recorded runs in runs/ yet")
    for s in streams:
        rep = B.check_stream(s)
        assert rep.ok, (s, rep.problems[:5])