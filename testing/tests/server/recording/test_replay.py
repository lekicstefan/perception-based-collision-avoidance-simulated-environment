"""Tests for the stream recorder, the replay tool and the offline processing mode."""
import numpy as np

from server.communication import protocol as P
from server.runtime.cycle import Processor
from server.recording.replay import ReplayLink, process_recording
from server.recording.stream_log import StreamRecorder, iter_records
from testing.support.session import Session, first_messages, drain_until
from testing.support.synthetic_stream import write_recording
from server.communication.transport import Link


class Collect(Processor):
    def __init__(self):
        self.batches = []

    def step(self, msgs, calibration):
        if msgs:
            self.batches.append([(m.header.type.name, m.header.seq) for m in msgs])
        return None


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
    drain_until(link, lambda: link.stats["lidar"].received >= 2 and link.stats["camera"].received >= 2 and link.stats["pose"].received >= 3)
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