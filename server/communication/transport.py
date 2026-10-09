"""Processor side of the ZeroMQ transport: docs/protocol.md sections 1 and 6.

Link owns the sockets (one SUB per Unity stream, one PUB for commands), performs the handshake, and hands the
processing cycle everything that arrived as Incoming(header, raw) objects. Session messages (HELLO, CALIBRATION,
END_OF_RUN) are consumed here. Per-stream sequence numbers are checked so lost messages are counted, not hidden.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import zmq

from server.communication import protocol as P

SOCKET_BUFFER_BYTES = 64 * 1024     # kernel buffers on the sensor sockets, bounds how stale queued data can get
HWM = {"session": 1000, "lidar": 5, "camera": 10, "pose": 100, "oracle": 5}
COMMAND_HWM = 10
DATA_STREAMS = ("lidar", "camera", "pose", "oracle")
EXPECTED = {
    "session": P.SESSION_TYPES,
    "lidar": (P.MsgType.LIDAR,),
    "camera": (P.MsgType.CAMERA,),
    "pose": (P.MsgType.POSE,),
    "oracle": (P.MsgType.ORACLE,),
}


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
    first_t: float | None = None
    last_t: float | None = None

    def update(self, seq: int, t: float) -> None:
        if self.first_seq is None:
            self.first_seq, self.first_t = seq, t
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
        self.last_t = t


@dataclass(frozen=True)
class Incoming:
    header: P.Header
    raw: bytes


class Link:
    def __init__(self, oracle: bool = False, host: str = "127.0.0.1", ports: dict | None = None,
                 heartbeat_timeout: float = 5.0):
        self.ports = {"session": P.PORT_SESSION, "command": P.PORT_COMMAND, "lidar": P.PORT_LIDAR,
                      "camera": P.PORT_CAMERA, "pose": P.PORT_POSE, "oracle": P.PORT_ORACLE}
        if ports:
            self.ports.update(ports)
        self.heartbeat_timeout = heartbeat_timeout
        self.ctx = zmq.Context()

        self.cmd = self.ctx.socket(zmq.PUB)
        self.cmd.setsockopt(zmq.SNDHWM, COMMAND_HWM)
        self.cmd.setsockopt(zmq.LINGER, 200)
        self.cmd.bind(f"tcp://{host}:{self.ports['command']}")

        names = ["session", "lidar", "camera", "pose"] + (["oracle"] if oracle else [])   # no oracle socket in normal mode
        self.subs: dict[str, zmq.Socket] = {}
        self.poller = zmq.Poller()
        for name in names:
            s = self.ctx.socket(zmq.SUB)
            s.setsockopt(zmq.RCVHWM, HWM[name])
            if name != "session":
                s.setsockopt(zmq.RCVBUF, SOCKET_BUFFER_BYTES)
            s.setsockopt(zmq.LINGER, 0)
            s.setsockopt(zmq.SUBSCRIBE, b"")
            s.connect(f"tcp://{host}:{self.ports[name]}")
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
        self._pending: list[Incoming] = []
        self._last_session = time.monotonic()
        self.recorder = None                 # StreamRecorder, see attach_recorder()
        self._rec_buffer: list | None = None # (arrival time, raw) kept during the handshake, before the run folder is known

    # ------------------------------------------------------------------ receiving

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
                self._record(raw)
            elif cid != self.calibration_id:
                raise P.ProtocolError(f"calibration id changed during the run ({self.calibration_id} -> {cid})")
        elif h.type == P.MsgType.END_OF_RUN and not self.ended:   # Unity sends several copies
            self.ended = True
            self.end_sim_time = data.get("sim_time")
            self._record(raw)

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
                try:
                    h = P.parse_header(raw)
                    if h.type not in EXPECTED[name]:
                        raise P.ProtocolError(f"{h.type.name} message on the {name} socket")
                    if name == "session":
                        self._handle_session(h, raw)
                        continue
                except P.ProtocolError as e:
                    if "calibration id changed" in str(e):
                        raise                       # a changed calibration invalidates the run
                    self._error(name, e)
                    continue
                self.stats[name].update(h.seq, h.t)
                self._record(raw)
                out.append(Incoming(h, raw))
        return out

    def _record(self, raw: bytes) -> None:
        if self.recorder is not None:
            self.recorder.write(raw)
        elif self._rec_buffer is not None:
            self._rec_buffer.append((time.monotonic(), raw))

    def attach_recorder(self, recorder) -> None:
        """Start recording. Messages that arrived during the handshake (buffer_for_recording=True) are written first."""
        for arrival, raw in self._rec_buffer or []:
            recorder.write(raw, arrival)
        self._rec_buffer = None
        self.recorder = recorder

    def stop_buffering(self) -> None:
        self._rec_buffer = None

    def drain(self, timeout_ms: int = 0) -> list[Incoming]:
        """Everything that arrived since the last call (sensor messages only), in arrival order per stream."""
        msgs = self._pending + self._receive(timeout_ms)
        self._pending = []
        return msgs

    def check_alive(self) -> None:
        """Call once per cycle. Unity resends CALIBRATION every second (also while paused) as a heartbeat."""
        if not self.ended and time.monotonic() - self._last_session > self.heartbeat_timeout:
            raise LinkTimeout(f"no session message from Unity for {self.heartbeat_timeout:.1f} s")

    # ------------------------------------------------------------------ handshake and sending

    def handshake(self, warmup=None, timeout: float = 120.0, settle: float = 0.3, ready_period: float = 0.1,
                  data_timeout: float = 10.0, buffer_for_recording: bool = False, run_dir: str | None = None) -> None:
        """Warm up, wait for HELLO, settle, then send READY until the first POSE message arrives.
        buffer_for_recording: keep what arrives during the handshake so attach_recorder() can write it afterwards."""
        if buffer_for_recording:
            self._rec_buffer = []
        if warmup is not None:
            warmup()                                # Numba compilation etc. before Unity is told to start
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
        while self.stats["pose"].received == 0:
            now = time.monotonic()
            if now >= next_ready:
                self.cmd.send(P.build_ready(run_dir))
                next_ready = now + ready_period
            self._pending += self._receive(20)
            if now - t_ready > data_timeout:
                raise LinkTimeout("READY sent but no data arrived from Unity")
        self._last_session = time.monotonic()

    def send_raw(self, raw: bytes) -> None:
        self.cmd.send(raw)

    def send_command(self, command: P.Command) -> None:
        self.cmd.send(P.build_command(command))

    def send_telemetry(self, obj: dict) -> None:
        self.cmd.send(P.build_telemetry(obj))

    # ------------------------------------------------------------------ reporting and shutdown

    def summary(self) -> str:
        lines = []
        for n, s in self.stats.items():
            gaps = ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in s.gaps[:5]) or "none"
            lines.append(f"{n:7s} received {s.received:6d}  first seq {s.first_seq}  lost {s.lost:5d}  "
                         f"out of order {s.out_of_order}  gaps: {gaps}")
        lines.append(f"calibration id {self.calibration_id}, end of run {self.ended} (sim time {self.end_sim_time}), "
                     f"{self.error_count} malformed messages")
        return "\n".join(lines)

    def close(self) -> None:
        """Safe to call more than once. destroy() closes every socket, so terminating the context cannot hang."""
        if not self.ctx.closed:
            self.ctx.destroy(linger=200)