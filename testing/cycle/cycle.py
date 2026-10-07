"""The processing cycle (docs: spec section 4.2). At a fixed rate it drains everything that arrived, orders it by
timestamp, lets the processor handle it once, and sends at most one command.

Timing: deadline = period (33 ms at 30 FPS). A cycle that takes longer than the period is an overrun. The runner
never tries to catch up with a burst of cycles: after a late cycle the next one simply starts right away.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

import protocol as P
from transport import Incoming, Link


class Processor:
    """What the real pipeline (perception, tracking, risk) will implement in later phases. This one does nothing."""

    def warmup(self) -> None:
        """Runs before the handshake completes (Numba compilation etc.), so the first live cycles meet the deadline."""

    def step(self, msgs: list[Incoming], calibration: dict) -> P.Command | None:
        """msgs: all new sensor messages, sorted by capture timestamp. Return a command, or None to send nothing."""
        return None


class CycleRunner:
    def __init__(self, link: Link, processor: Processor, rate_hz: float = 30.0):
        self.link = link
        self.processor = processor
        self.period = 1.0 / rate_hz
        self.rate_hz = rate_hz
        self.drain_ms: list[float] = []      # receive and parse
        self.process_ms: list[float] = []    # processor.step
        self.total_ms: list[float] = []      # whole cycle including sending
        self.counts: list[int] = []          # messages handled per cycle
        self.overruns = 0
        self.duration = 0.0

    def run(self, max_seconds: float | None = None) -> None:
        link = self.link
        held: list[Incoming] = []            # data that arrived before the calibration did
        t_begin = time.perf_counter()
        next_tick = t_begin
        while not link.ended:
            if max_seconds is not None and time.perf_counter() - t_begin > max_seconds:
                break
            wait = next_tick - time.perf_counter()
            if wait > 0:
                time.sleep(wait)

            t0 = time.perf_counter()
            link.check_alive()
            msgs = held + link.drain(0)
            held = []
            t1 = time.perf_counter()

            cmd = None
            if link.calibration is None:     # arrives right after READY, normally within a few milliseconds
                held, msgs = msgs, []
            else:
                msgs.sort(key=lambda m: m.header.t)          # stable: same timestamp keeps arrival order
                cmd = self.processor.step(msgs, link.calibration)
            t2 = time.perf_counter()

            if cmd is not None:
                link.send_command(dataclasses.replace(cmd, processing_time=t2 - t1))
            t3 = time.perf_counter()

            self.drain_ms.append((t1 - t0) * 1e3)
            self.process_ms.append((t2 - t1) * 1e3)
            self.total_ms.append((t3 - t0) * 1e3)
            self.counts.append(len(msgs))
            if t3 - t0 > self.period:
                self.overruns += 1

            next_tick += self.period
            if next_tick < t3:               # late: no catch-up burst
                next_tick = t3
        self.duration = time.perf_counter() - t_begin

    def report(self) -> str:
        n = len(self.total_ms)
        if n == 0:
            return "no cycles ran"

        def row(name: str, v: list[float]) -> str:
            a = np.asarray(v)
            return (f"{name:8s} mean {a.mean():7.3f}  p50 {np.percentile(a, 50):7.3f}  p95 {np.percentile(a, 95):7.3f}  "
                    f"p99 {np.percentile(a, 99):7.3f}  max {a.max():7.3f}")

        return "\n".join([
            f"{n} cycles in {self.duration:.1f} s ({n / max(self.duration, 1e-9):.1f} per second, target {self.rate_hz:g}), "
            f"deadline {self.period * 1e3:.1f} ms",
            f"messages per cycle: mean {np.mean(self.counts):.2f}, max {max(self.counts)}",
            "latency in ms:",
            "  " + row("receive", self.drain_ms),
            "  " + row("process", self.process_ms),
            "  " + row("total", self.total_ms),
            f"overruns (total > deadline): {self.overruns} ({100.0 * self.overruns / n:.1f} %)",
        ])