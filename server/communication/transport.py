from __future__ import annotations

import time
from dataclasses import dataclass, field

import zmq

from server.communication import protocol as P

SOCKET_BUFFER_BYTES = 64 * 1024     # kernel buffers on the bundle socket, bounds how stale queued data can get
HWM = {"session": 1000, "bundle": 3, "oracle": 5}
COMMAND_HWM = 10
EXPECTED = {"session": P.SESSION_TYPES, "bundle": (P.MsgType.BUNDLE,), "oracle": (P.MsgType.ORACLE,)}


class LinkTimeout(RuntimeError):
    """Unity never answered, or stopped sending session messages (heartbeat) for too long."""


@dataclass
class StreamStats:
    name: str
    received: int = 0
    first_seq: int | None = None
    next_seq: int = 0
    lost: int = 0
    out_of_order: int = 0
    gaps: list = field(default_factory=list)      # (first_lost_seq, last_lost_seq)

    def update(self, seq: int) -> None:
        if self.first_seq is None:
            self.first_seq = seq
            if seq != 0:                           # slow joiner: the first messages never arrived
                self.lost += seq
                self.gaps.append((0, seq - 1))
            self.next_seq = seq + 1
        elif seq == self.next_seq:
            self.next_seq += 1
        elif seq > self.next_seq:
            self.lost += seq - self.next_seq
            self.gaps.append((self.next_seq, seq - 1))
            self.next_seq = seq + 1
        else:
            self.out_of_order += 1
        self.received += 1


@dataclass(frozen=True)
class Incoming:
    recv_ns: int
    header: P.Header
    raw: bytes


class Link:
    def __init__(self, oracle: bool = False, host: str = "127.0.0.1", heartbeat_timeout: float = 5.0):
        self.heartbeat_timeout = heartbeat_timeout
        self.ctx = zmq.Context()
        self.cmd = self.ctx.socket(zmq.PUB)
        self.cmd.setsockopt(zmq.SNDHWM, COMMAND_HWM)
        self.cmd.setsockopt(zmq.LINGER, 200)
        self._bind_with_retry(self.cmd, f"tcp://{host}:{P.PORT_COMMAND}")

        ports = {"session": P.PORT_SESSION, "bundle": P.PORT_BUNDLE, "oracle": P.PORT_ORACLE}
        names = ["session", "bundle"] + (["oracle"] if oracle else [])
        self.subs: dict[str, zmq.Socket] = {}
        self.poller = zmq.Poller()
        for name in names:
            s = self.ctx.socket(zmq.SUB)
            s.setsockopt(zmq.RCVHWM, HWM[name])
            if name != "session":
                s.setsockopt(zmq.RCVBUF, SOCKET_BUFFER_BYTES)
            s.setsockopt(zmq.LINGER, 0)
            s.setsockopt(zmq.SUBSCRIBE, b"")
            s.connect(f"tcp://{host}:{ports[name]}")
            self.subs[name] = s
            self.poller.register(s, zmq.POLLIN)

        self.stats = {n: StreamStats(n) for n in names if n != "session"}
        self.hello: dict | None = None
        self.calibration: dict | None = None
        self.calibration_id: str | None = None
        self.ended = False
        self.end_sim_time: float | None = None
        self.errors: list[str] = []
        self.error_count = 0
        self.ignored_before_hello = 0        # leftovers of an earlier Unity run, see _receive
        self._pending: list[Incoming] = []
        self._last_session = time.monotonic()

    @staticmethod
    def _bind_with_retry(sock, address: str, timeout: float = 10.0) -> None:
        """The previous run may still be closing its socket (--loop): a busy port is retried for a while."""
        t0 = time.monotonic()
        while True:
            try:
                sock.bind(address)
                return
            except zmq.ZMQError as e:
                if e.errno != zmq.EADDRINUSE or time.monotonic() - t0 > timeout:
                    raise
                time.sleep(0.2)

    def _error(self, where: str, e: Exception) -> None:
        self.error_count += 1
        if len(self.errors) < 50:
            self.errors.append(f"{where}: {e}")

    def _handle_session(self, h: P.Header, raw: bytes) -> None:
        self._last_session = time.monotonic()
        _, data = P.parse_session(raw)
        if h.type == P.MsgType.HELLO:
            if self.hello is None:
                self.hello = data
        elif h.type == P.MsgType.CALIBRATION:
            cid = data.get("calibrationId")
            if self.calibration is None:
                self.calibration, self.calibration_id = data, cid
            elif cid != self.calibration_id:
                raise P.ProtocolError(f"calibration id changed during the run ({self.calibration_id} -> {cid})")
        elif h.type == P.MsgType.END_OF_RUN and not self.ended:   # Unity sends several copies
            self.ended = True
            self.end_sim_time = data.get("sim_time")

    def _receive(self, timeout_ms: int) -> list[Incoming]:
        out: list[Incoming] = []
        ready = dict(self.poller.poll(timeout_ms))
        for name, sock in self.subs.items():
            if sock not in ready:
                continue
            while True:
                try:
                    raw = sock.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                recv_ns = time.perf_counter_ns()
                try:
                    h = P.parse_header(raw)
                    if h.type not in EXPECTED[name]:
                        raise P.ProtocolError(f"{h.type.name} message on the {name} socket")
                    if self.hello is None and not (name == "session" and h.type == P.MsgType.HELLO):
                        # A run always begins with HELLO and no data flows before READY: anything earlier belongs to a
                        # Unity run that is already over (stale END_OF_RUN copies) and must not end or feed this run.
                        self.ignored_before_hello += 1
                        continue
                    if name == "session":
                        self._handle_session(h, raw)
                        continue
                except P.ProtocolError as e:
                    if "calibration id changed" in str(e):
                        raise
                    self._error(name, e)
                    continue
                self.stats[name].update(h.seq)
                out.append(Incoming(recv_ns, h, raw))
        return out

    def receive(self, timeout_ms: int = 0) -> list[Incoming]:
        """Waits up to timeout_ms for something, returns every bundle (and oracle message) that has arrived, in arrival order."""
        msgs = self._pending + self._receive(timeout_ms)
        self._pending = []
        return msgs

    def check_alive(self) -> None:
        """Unity resends CALIBRATION every second (also while paused) as a heartbeat."""
        if not self.ended and time.monotonic() - self._last_session > self.heartbeat_timeout:
            raise LinkTimeout(f"no session message from Unity for {self.heartbeat_timeout:.1f} s")

    def handshake(self, warmup=None, timeout: float = 120.0, settle: float = 0.3, ready_period: float = 0.1,
                  data_timeout: float = 10.0, run_dir: str | None = None) -> None:
        """Warm up, wait for HELLO, settle, then send READY until the first bundle arrives."""
        if warmup is not None:
            warmup()
        t0 = time.monotonic()
        while self.hello is None:
            self._pending += self._receive(100)
            if time.monotonic() - t0 > timeout:
                raise LinkTimeout("no HELLO from Unity")
        if self.hello.get("schema") != P.VERSION:
            raise P.ProtocolError(f"Unity speaks schema {self.hello.get('schema')}, expected {P.VERSION}")
        self._pending += self._receive(int(settle * 1000))
        t_ready = time.monotonic()
        next_ready = 0.0
        while self.stats["bundle"].received == 0:
            now = time.monotonic()
            if now >= next_ready:
                self.cmd.send(P.build_ready(run_dir))
                next_ready = now + ready_period
            self._pending += self._receive(20)
            if now - t_ready > data_timeout:
                raise LinkTimeout("READY sent but no data arrived from Unity")
        self._last_session = time.monotonic()

    def send_command(self, command: P.Command) -> None:
        self.cmd.send(P.build_command(command))

    def send_telemetry(self, obj: dict) -> None:
        self.cmd.send(P.build_telemetry(obj))

    def summary(self) -> str:
        lines = []
        for n, s in self.stats.items():
            gaps = ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in s.gaps[:5]) or "none"
            lines.append(f"{n:7s} received {s.received:6d}  first seq {s.first_seq}  lost {s.lost:5d}  "
                         f"out of order {s.out_of_order}  gaps: {gaps}")
        lines.append(f"calibration id {self.calibration_id}, end of run {self.ended} (sim time {self.end_sim_time}), "
                     f"{self.error_count} malformed messages")
        if self.ignored_before_hello:
            lines.append(f"{self.ignored_before_hello} messages ignored because they arrived before HELLO (left over from an earlier Unity run)")
        return "\n".join(lines)

    def close(self) -> None:
        if not self.ctx.closed:
            self.ctx.destroy(linger=200)