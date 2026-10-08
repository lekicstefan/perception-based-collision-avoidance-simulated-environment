"""Transport tests: the processor Link against a fake Unity (pyzmq), no simulator needed."""
import itertools
import threading
import time

import numpy as np
import pytest

import protocol as P
from fake_unity import FakeUnity
from transport import Link, LinkTimeout

_port = itertools.count(26000, 10)


@pytest.fixture
def ports():
    base = next(_port)
    return {"session": base, "command": base + 1, "lidar": base + 2, "camera": base + 3, "pose": base + 4, "oracle": base + 5}


@pytest.fixture(autouse=True)
def cleanup(monkeypatch):
    """Close every Link and FakeUnity a test created, also when it fails (an open context blocks the exit)."""
    created = []
    orig_link, orig_fake = Link.__init__, FakeUnity.__init__

    def link_init(self, *a, **k):
        orig_link(self, *a, **k)
        created.append(self)

    def fake_init(self, *a, **k):
        orig_fake(self, *a, **k)
        created.append(self)

    monkeypatch.setattr(Link, "__init__", link_init)
    monkeypatch.setattr(FakeUnity, "__init__", fake_init)
    yield
    for obj in created:
        obj.close()


class Session:
    """Runs the fake Unity in a thread: wait for READY, then run `script(fake)`."""

    def __init__(self, ports, script=None, oracle=False, calibration=None):
        self.fake = FakeUnity(oracle=oracle, ports=ports, calibration=calibration)
        self.script = script
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            self.fake.wait_ready()
            if self.script:
                self.script(self.fake)
        except Exception as e:      # reported by the test through .error
            self.error = e

    def finish(self):
        self.thread.join(10)
        assert self.error is None, self.error
        self.fake.close()


def first_messages(fake):
    for i in range(3):
        fake.send_pose(0.02 * i)
    for i in range(2):
        fake.send_lidar(0.1 * i)
        fake.send_camera(0.033 * i)


def test_handshake_and_streams(ports):
    s = Session(ports, first_messages)
    link = Link(ports=ports)
    link.handshake(settle=0.1)
    msgs = link.drain(300)
    s.finish()
    assert link.calibration_id == "fake-calibration"
    assert {m.header.type for m in msgs} <= {P.MsgType.POSE, P.MsgType.LIDAR, P.MsgType.CAMERA}
    for name, count in (("pose", 3), ("lidar", 2), ("camera", 2)):
        st = link.stats[name]
        assert st.first_seq == 0 and st.lost == 0 and st.received == count, (name, st)
    assert link.error_count == 0
    link.close()


def test_messages_before_the_first_drain_are_not_lost(ports):
    s = Session(ports, first_messages)
    link = Link(ports=ports)
    link.handshake(settle=0.1)             # data arrives while the READY loop runs, it must be kept
    time.sleep(0.2)
    types = [m.header.type for m in link.drain(100)]
    s.finish()
    assert types.count(P.MsgType.POSE) == 3
    link.close()


def test_command_reaches_unity(ports):
    s = Session(ports, lambda f: (first_messages(f), time.sleep(0.5), f.poll_commands()))
    link = Link(ports=ports)
    link.handshake(settle=0.1)
    cmd = P.Command(42, 1.48, 12.7, 17.5, 3.0, 0.0, 2.5, 0.2, 0.012, False, 1)
    for _ in range(5):                      # PUB/SUB, a few copies like the real 30 Hz stream
        link.send_command(cmd)
        time.sleep(0.05)
    link.send_telemetry({"speed": 1.0})
    s.finish()
    types = [t for t, _ in s.fake.commands]
    assert P.CmdType.COMMAND in types
    got = next(o for t, o in s.fake.commands if t == P.CmdType.COMMAND)
    assert got.id == 42 and got.emergency is False and got.risk_level == 1
    link.close()


def test_end_of_run_is_deduplicated(ports):
    s = Session(ports, lambda f: (first_messages(f), f.end_of_run(12.34, copies=5, period=0.02)))
    link = Link(ports=ports)
    link.handshake(settle=0.1)
    time.sleep(0.4)
    link.drain(100)
    s.finish()
    assert link.ended and link.end_sim_time == 12.34
    link.check_alive()                       # an ended run never times out
    link.close()


def test_calibration_resend_is_ignored_but_a_change_is_an_error(ports):
    def script(f):
        first_messages(f)
        f.send_calibration()                                   # same id again: heartbeat
        f.calibration = {"calibrationId": "something-else"}
        f.send_calibration()
    s = Session(ports, script)
    link = Link(ports=ports)
    with pytest.raises(P.ProtocolError, match="calibration id changed"):
        link.handshake(settle=0.1)          # the changed calibration may already arrive during the handshake
        for _ in range(10):
            link.drain(100)
    s.finish()


def test_heartbeat_timeout(ports):
    s = Session(ports, first_messages)
    link = Link(ports=ports, heartbeat_timeout=0.3)
    link.handshake(settle=0.1)
    s.finish()
    link.check_alive()                       # fresh
    time.sleep(0.45)
    with pytest.raises(LinkTimeout):
        link.check_alive()
    link.close()


def test_missing_sequence_numbers_are_counted(ports):
    def script(f):
        first_messages(f)
        f.seq["pose"] += 4                    # four pose messages never arrive
        f.send_pose(0.1)
        f.send_pose(0.12)
    s = Session(ports, script)
    link = Link(ports=ports)
    link.handshake(settle=0.1)
    link.drain(300)
    s.finish()
    st = link.stats["pose"]
    assert st.lost == 4 and st.gaps == [(3, 6)] and st.received == 5
    link.close()


def test_a_message_of_the_wrong_type_is_dropped_and_counted(ports):
    def script(f):
        first_messages(f)
        f.pubs["pose"].send(P.build_message(P.MsgType.LIDAR, 99, P.lidar_payload(np.zeros((2, 2), np.uint16))))
        f.pubs["pose"].send(b"garbage")
    s = Session(ports, script)
    link = Link(ports=ports)
    link.handshake(settle=0.1)
    msgs = link.drain(300)
    s.finish()
    assert link.error_count == 2 and len(link.errors) == 2
    assert all(m.header.type != P.MsgType.LIDAR or m.header.seq < 99 for m in msgs)
    assert link.stats["pose"].received == 3
    link.close()


def test_no_oracle_socket_in_normal_mode(ports):
    link = Link(ports=ports)
    assert "oracle" not in link.subs and "oracle" not in link.stats
    link.close()


def test_oracle_stream_when_requested(ports):
    objs = np.zeros(2, dtype=P.ORACLE_DTYPE)
    objs["id"] = [1, 2]

    def script(f):
        first_messages(f)
        f.send_oracle(0.0, objs)
    s = Session(ports, script, oracle=True)
    link = Link(ports=ports, oracle=True)
    link.handshake(settle=0.1)
    msgs = link.drain(300)
    s.finish()
    oracle = [m for m in msgs if m.header.type == P.MsgType.ORACLE]
    assert len(oracle) == 1 and len(P.parse_oracle(oracle[0].raw)[1]) == 2
    link.close()


def test_handshake_times_out_without_unity(ports):
    link = Link(ports=ports)
    with pytest.raises(LinkTimeout):
        link.handshake(timeout=0.4)
    link.close()


def test_warmup_runs_before_ready(ports):
    order = []
    s = Session(ports, first_messages)
    link = Link(ports=ports)
    link.handshake(warmup=lambda: order.append("warmup"), settle=0.1)
    order.append("ready")
    s.finish()
    assert order == ["warmup", "ready"]
    link.close()