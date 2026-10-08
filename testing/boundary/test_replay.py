"""Tests for the stream recorder, the replay tool and the offline processing mode."""
import numpy as np

import protocol as P
from cycle import Processor
from replay import ReplayLink, process_recording
from testing.replay.stream_log import StreamRecorder, iter_records
from test_transport import Session, cleanup, first_messages, ports  # noqa: F401  (fixtures)
from transport import Link


class Collect(Processor):
    def __init__(self):
        self.batches = []

    def step(self, msgs, calibration):
        if msgs:
            self.batches.append([(m.header.type.name, m.header.seq) for m in msgs])
        return None


def write_recording(path, seconds=1.0):
    """1 s at the default rates: pose 50 Hz, camera 30 Hz, LiDAR 10 Hz, plus calibration and end of run."""
    rec = StreamRecorder(path)
    rec.write(P.build_message(P.MsgType.CALIBRATION, 0, P.json_payload({"calibrationId": "synthetic"})), 0.0)
    seq = {"pose": 0, "camera": 0, "lidar": 0}
    cam = P.camera_payload(4, 2, P.CameraFormat.RAW, bytes(24))
    lid = P.lidar_payload(np.full((4, 8), 5000, dtype=np.uint16))
    for i in range(int(seconds * 100)):
        t = i * 0.01
        if i % 2 == 0:
            rec.write(P.build_message(P.MsgType.POSE, seq["pose"], t=t), t); seq["pose"] += 1
        if i % 3 == 0:
            rec.write(P.build_message(P.MsgType.CAMERA, seq["camera"], cam, t=t), t); seq["camera"] += 1
        if i % 10 == 0:
            rec.write(P.build_message(P.MsgType.LIDAR, seq["lidar"], lid, t=t), t); seq["lidar"] += 1
    rec.write(P.build_message(P.MsgType.END_OF_RUN, 0, P.json_payload({"sim_time": seconds})), seconds)
    rec.close()
    return seq


def test_recorder_round_trip_and_a_cut_off_file(tmp_path):
    a, b, c = b"first", b"second message", b"third"
    rec = StreamRecorder(tmp_path / "s.bin")
    rec.write(a, 10.0); rec.write(b, 10.5); rec.write(c, 11.0)
    rec.close()
    got = list(iter_records(tmp_path / "s.bin"))
    assert [raw for _, raw in got] == [a, b, c]
    assert [round(t, 3) for t, _ in got] == [0.0, 0.5, 1.0]
    data = (tmp_path / "s.bin").read_bytes()
    (tmp_path / "cut.bin").write_bytes(data[:-3])                 # crash in the middle of the last record
    assert [raw for _, raw in iter_records(tmp_path / "cut.bin")] == [a, b]


def test_replay_as_fast_as_possible_is_deterministic(tmp_path):
    seq = write_recording(tmp_path / "stream.bin")
    first, second = Collect(), Collect()
    _, link = process_recording(tmp_path, first, rate_hz=30, speed=0)
    process_recording(tmp_path, second, rate_hz=30, speed=0)
    assert first.batches == second.batches
    assert sum(len(b) for b in first.batches) == seq["pose"] + seq["camera"] + seq["lidar"] == 94
    assert link.ended and link.calibration_id == "synthetic" and link.end_sim_time == 1.0
    assert all(s.first_seq == 0 and s.lost == 0 for s in link.stats.values())


def test_replay_in_original_timing_can_be_sped_up(tmp_path):
    write_recording(tmp_path / "stream.bin")
    col = Collect()
    runner, link = process_recording(tmp_path, col, rate_hz=30, speed=20.0)    # 1 s of data in about 50 ms
    assert sum(len(b) for b in col.batches) == 94 and link.ended
    assert runner.duration < 3.0


def test_live_recording_includes_what_arrived_during_the_handshake(tmp_path, ports):
    s = Session(ports, first_messages)
    link = Link(ports=ports)
    link.handshake(settle=0.1, buffer_for_recording=True)
    rec = StreamRecorder(tmp_path / "live.bin")
    link.attach_recorder(rec)
    link.drain(300)
    s.finish()
    rec.close()
    types = [P.parse_header(raw).type for _, raw in iter_records(tmp_path / "live.bin")]
    assert types.count(P.MsgType.CALIBRATION) == 1
    assert types.count(P.MsgType.POSE) == 3 and types.count(P.MsgType.LIDAR) == 2 and types.count(P.MsgType.CAMERA) == 2
    replayed = ReplayLink(tmp_path / "live.bin", speed=0)
    got = []
    while not replayed.ended:
        got += replayed.drain(0)
    assert len(got) == 7 and replayed.stats["pose"].first_seq == 0