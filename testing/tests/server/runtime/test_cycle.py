"""Tests for the processing cycle, using the fake Unity (no simulator needed)."""
import itertools
import threading
import time

from server.communication import protocol as P
from server.runtime.cycle import CycleRunner, Processor
from testing.support.fake_unity import FakeUnity
from server.communication.transport import Link

_base = itertools.count(26900, 10)


class Recorder(Processor):
    def __init__(self, step_seconds: float = 0.0):
        self.batches = []
        self.step_seconds = step_seconds

    def step(self, msgs, calibration):
        if msgs:
            self.batches.append([m.header.t for m in msgs])
        if self.step_seconds:
            time.sleep(self.step_seconds)
        return None


def run_cycles(processor, rate_hz):
    b = next(_base)
    ports = {"session": b, "command": b + 1, "lidar": b + 2, "camera": b + 3, "pose": b + 4, "oracle": b + 5}
    fake, link = FakeUnity(ports=ports), Link(ports=ports)

    def unity():
        fake.wait_ready()
        fake.send_lidar(0.30)           # sent out of timestamp order on purpose
        fake.send_pose(0.10)
        fake.send_camera(0.20)
        time.sleep(0.3)
        fake.end_of_run(1.0, copies=3, period=0.02)

    thread = threading.Thread(target=unity, daemon=True)
    thread.start()
    runner = CycleRunner(link, processor, rate_hz=rate_hz)
    try:
        link.handshake(settle=0.1)
        runner.run(max_seconds=10)
    finally:
        thread.join(5)
        link.close()
        fake.close()
    return runner, link


def test_messages_are_ordered_by_timestamp_and_the_run_ends():
    rec = Recorder()
    runner, link = run_cycles(rec, rate_hz=50)
    assert link.ended
    assert all(batch == sorted(batch) for batch in rec.batches)
    assert sorted(t for batch in rec.batches for t in batch) == [0.10, 0.20, 0.30]
    assert len(runner.total_ms) > 5 and runner.overruns == 0


def test_a_slow_processor_counts_overruns_and_does_not_burst():
    runner, _ = run_cycles(Recorder(step_seconds=0.03), rate_hz=100)    # 30 ms of work, 10 ms deadline
    assert runner.overruns >= len(runner.total_ms) - 1
    assert len(runner.total_ms) < 0.45 / 0.03 + 5                       # roughly one cycle per 30 ms, no catch-up burst