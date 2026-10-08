"""Replay of a recorded sensor stream (step 4.8), and the offline processing mode.

ReplayLink stands in for Link: the CycleRunner and the processor cannot tell the difference.
  speed 1.0   original timing (messages are delivered when they originally arrived); 2.0 = twice as fast
  speed 0     as fast as possible: no waiting, every cycle receives one cycle period of recorded time. This is
              deterministic, so the same recording always gives the same sequence of batches (tests, debugging).
"""
from __future__ import annotations

import time

import protocol as P
from cycle import CycleRunner, Processor
from stream_log import iter_records, resolve_stream_path
from transport import Incoming, Link, StreamStats

KIND = {P.MsgType.LIDAR: "lidar", P.MsgType.CAMERA: "camera", P.MsgType.POSE: "pose", P.MsgType.ORACLE: "oracle"}


class ReplayLink:
    summary = Link.summary                    # same report as the live link

    def __init__(self, path, speed: float = 1.0, period: float = 1.0 / 30.0):
        self.path = resolve_stream_path(path)
        self.speed = speed
        self.period = period                  # recorded time per cycle in as-fast-as-possible mode
        self._records = iter_records(self.path)
        self._next = next(self._records, None)
        self._t0 = time.monotonic()
        self._virtual = 0.0
        self.stats: dict[str, StreamStats] = {}
        self.hello: dict = {}
        self.calibration: dict | None = None
        self.calibration_id: str | None = None
        self.ended = False
        self.end_sim_time: float | None = None
        self.errors: list[str] = []
        self.error_count = 0
        self.sent_commands: list[P.Command] = []

    def handshake(self, warmup=None, **_ignored) -> None:
        if warmup is not None:
            warmup()
        self._t0 = time.monotonic()

    def _now(self) -> float:
        if self.speed and self.speed > 0:
            return (time.monotonic() - self._t0) * self.speed
        self._virtual += self.period
        return self._virtual

    def _deliver(self, raw: bytes, out: list) -> None:
        try:
            h = P.parse_header(raw)
            if h.type == P.MsgType.CALIBRATION:
                self.calibration = P.parse_session(raw)[1]
                self.calibration_id = self.calibration.get("calibrationId")
                return
            if h.type == P.MsgType.END_OF_RUN:
                self.end_sim_time = P.parse_session(raw)[1].get("sim_time")
                return
            if h.type not in KIND:
                return
        except P.ProtocolError as e:
            self.error_count += 1
            if len(self.errors) < 50:
                self.errors.append(f"replay: {e}")
            return
        name = KIND[h.type]
        self.stats.setdefault(name, StreamStats(name)).update(h.seq, h.t)
        out.append(Incoming(h, raw))

    def drain(self, timeout_ms: int = 0) -> list[Incoming]:
        now = self._now()
        out: list[Incoming] = []
        while self._next is not None and self._next[0] <= now:
            raw = self._next[1]
            self._next = next(self._records, None)
            self._deliver(raw, out)
        if self._next is None:
            self.ended = True                 # everything has been delivered
        return out

    def check_alive(self) -> None:
        pass

    def send_command(self, command: P.Command) -> None:
        self.sent_commands.append(command)    # nobody is listening; kept so tests can inspect them

    def send_telemetry(self, obj: dict) -> None:
        pass

    def close(self) -> None:
        pass


def process_recording(path, processor: Processor, rate_hz: float = 30.0, speed: float = 0.0, run_log=None):
    """Offline processing mode: runs a processor over a recording. Returns (runner, link)."""
    link = ReplayLink(path, speed=speed, period=1.0 / rate_hz)
    link.handshake(warmup=processor.warmup)
    runner = CycleRunner(link, processor, rate_hz=rate_hz, run_log=run_log, realtime=speed > 0)
    runner.run()
    return runner, link