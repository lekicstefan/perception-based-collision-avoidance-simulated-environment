"""Load shedding (server/runtime/shedding.py): the pure rule, and the cycle runner with a burst of scans and a slow processor."""
import itertools
import threading
import time
from types import SimpleNamespace

from server.communication import protocol as P
from server.communication.transport import Link
from server.recording.replay import process_recording
from server.runtime.cycle import CycleRunner, Processor
from server.runtime.shedding import SheddingConfig, shed
from testing.support.fake_unity import FakeUnity
from testing.support.synthetic_stream import write_recording

_base = itertools.count(27400, 10)


def msg(kind, t):
    return SimpleNamespace(header=SimpleNamespace(type=kind, t=t))


def burst(n_scans):
    """n scans 0.1 s apart, a camera frame every 1/30 s and a pose every 0.02 s, sorted by time."""
    ms = [msg(P.MsgType.LIDAR, round(0.1 * i, 6)) for i in range(n_scans)]
    ms += [msg(P.MsgType.CAMERA, round(i / 30.0, 6)) for i in range(3 * n_scans)]
    ms += [msg(P.MsgType.POSE, round(0.02 * i, 6)) for i in range(5 * n_scans)]
    return sorted(ms, key=lambda m: m.header.t)


def kinds(ms, kind):
    return [m.header.t for m in ms if m.header.type == kind]


def test_nothing_is_skipped_up_to_the_limit():
    ms = burst(2)
    kept, scans, cams = shed(ms, SheddingConfig(max_scans_per_cycle=2))
    assert kept == ms and scans == 0 and cams == 0


def test_the_newest_scans_are_kept_and_the_older_ones_skipped():
    ms = burst(5)
    kept, scans, cams = shed(ms, SheddingConfig(max_scans_per_cycle=2))
    assert kinds(kept, P.MsgType.LIDAR) == [0.3, 0.4] and scans == 3


def test_camera_frames_before_the_oldest_kept_scan_are_skipped_with_them():
    ms = burst(5)
    kept, scans, cams = shed(ms, SheddingConfig(max_scans_per_cycle=2))
    assert min(kinds(kept, P.MsgType.CAMERA)) >= 0.3 and cams == len([t for t in kinds(ms, P.MsgType.CAMERA) if t < 0.3])


def test_pose_messages_are_never_skipped_and_the_order_is_kept():
    ms = burst(5)
    kept, _, _ = shed(ms, SheddingConfig(max_scans_per_cycle=1))
    assert kinds(kept, P.MsgType.POSE) == kinds(ms, P.MsgType.POSE)
    assert [m.header.t for m in kept] == sorted(m.header.t for m in kept)


def test_zero_switches_shedding_off():
    ms = burst(8)
    kept, scans, cams = shed(ms, SheddingConfig(max_scans_per_cycle=0))
    assert kept == ms and scans == 0 and cams == 0


class SlowRecorder(Processor):
    def __init__(self, step_seconds):
        self.step_seconds, self.batches = step_seconds, []

    def step(self, msgs, calibration):
        self.batches.append([(m.header.type, m.header.t) for m in msgs])
        time.sleep(self.step_seconds)
        return None


def test_a_burst_of_scans_is_bounded_in_the_cycle_runner():
    b = next(_base)
    ports = {"session": b, "command": b + 1, "lidar": b + 2, "camera": b + 3, "pose": b + 4, "oracle": b + 5}
    fake, link = FakeUnity(ports=ports), Link(ports=ports)

    def unity():
        fake.wait_ready()
        for i in range(6):
            fake.send_lidar(0.1 * (i + 1))
        fake.send_pose(0.7)
        time.sleep(0.4)
        fake.end_of_run(1.0, copies=3, period=0.02)

    thread = threading.Thread(target=unity, daemon=True)
    thread.start()
    rec = SlowRecorder(0.05)
    runner = CycleRunner(link, rec, rate_hz=30, shedding=SheddingConfig(max_scans_per_cycle=2))
    try:
        link.handshake(settle=0.1)
        runner.run(max_seconds=10)
    finally:
        thread.join(5)
        link.close()
        fake.close()
    received = link.stats["lidar"].received                 # not all 6 need to arrive: the first message of a new connection can be lost (slow joiner)
    processed = sum(sum(1 for typ, _ in batch if typ == P.MsgType.LIDAR) for batch in rec.batches)
    assert received >= 4
    assert all(sum(1 for typ, _ in batch if typ == P.MsgType.LIDAR) <= 2 for batch in rec.batches)
    assert processed + runner.scans_dropped == received      # every scan that arrived was either processed or counted as skipped
    assert runner.scans_dropped >= 1
    assert "load shedding (at most 2 scans per cycle)" in runner.report()


def test_offline_processing_skips_nothing_unless_it_is_asked_to(tmp_path):
    write_recording(tmp_path / "stream.bin")                       # 1 s of data, replayed in about 50 ms: a burst, not an overload
    plain, _ = process_recording(tmp_path, SlowRecorder(0.0), rate_hz=30, speed=20.0)
    assert plain.scans_dropped == 0 and plain.camera_dropped == 0
    live_like, _ = process_recording(tmp_path, SlowRecorder(0.0), rate_hz=30, speed=20.0, shedding=SheddingConfig(max_scans_per_cycle=1))
    assert live_like.scans_dropped > 0