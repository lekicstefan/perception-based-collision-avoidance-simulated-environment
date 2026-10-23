from __future__ import annotations

import dataclasses
import time

import numpy as np

from server.communication import protocol as P
from server.communication.transport import Incoming, Link
from server.runtime.shedding import SheddingConfig, shed

CYCLE_COLUMNS = ["cycle", "wall_s", "newest_t", "n_msgs", "n_lidar", "n_camera", "n_pose",
                 "receive_ms", "process_ms", "total_ms", "overrun", "dropped_lidar", "dropped_camera"]
COMMAND_COLUMNS = ["cycle", "wall_s"] + [f.name for f in dataclasses.fields(P.Command)]


class Processor:
    """What the real pipeline (perception, tracking, risk) will implement in later phases. This one does nothing."""

    def warmup(self) -> None:
        """Runs before the handshake completes (Numba compilation etc.), so the first live cycles meet the deadline."""

    def step(self, msgs: list[Incoming], calibration: dict) -> P.Command | None:
        """msgs: all new sensor messages, sorted by capture timestamp. Return a command, or None to send nothing."""
        return None


class CycleRunner:
    def __init__(self, link: Link, processor: Processor, rate_hz: float = 30.0, run_log=None, realtime: bool = True,
                 shedding: SheddingConfig | None = None):
        self.link = link
        self.shedding = shedding if shedding is not None else SheddingConfig()
        self.processor = processor
        self.period = 1.0 / rate_hz
        self.rate_hz = rate_hz
        self.realtime = realtime             # False: replay as fast as possible, no waiting between cycles
        self.drain_ms: list[float] = []      # receive and parse
        self.process_ms: list[float] = []    # processor.step
        self.total_ms: list[float] = []      # whole cycle including sending
        self.counts: list[int] = []          # messages handled per cycle
        self.overruns = 0
        self.scans_dropped = 0               # skipped by the load shedding (shedding.py)
        self.camera_dropped = 0
        self.cycles_shedding = 0             # cycles in which anything was skipped
        self.duration = 0.0
        self.cycle_table = run_log.table("cycles", CYCLE_COLUMNS) if run_log else None
        self.command_table = run_log.table("commands", COMMAND_COLUMNS) if run_log else None

    def run(self, max_seconds: float | None = None) -> None:
        link = self.link
        held: list[Incoming] = []            # data that arrived before the calibration did
        t_begin = time.perf_counter()
        next_tick = t_begin
        cycle = 0
        while not link.ended:
            if max_seconds is not None and time.perf_counter() - t_begin > max_seconds:
                break
            if self.realtime:
                wait = next_tick - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)

            t0 = time.perf_counter()
            link.check_alive()
            msgs = held + link.drain(0)
            held = []
            t1 = time.perf_counter()

            cmd = None
            arrived, n_scans_skipped, n_camera_skipped = msgs, 0, 0
            if link.calibration is None:     # arrives right after READY, normally within a few milliseconds
                held, msgs = msgs, []
            else:
                msgs.sort(key=lambda m: m.header.t)          # stable: same timestamp keeps arrival order
                arrived = msgs
                msgs, n_scans_skipped, n_camera_skipped = shed(msgs, self.shedding)
                cmd = self.processor.step(msgs, link.calibration)
            t2 = time.perf_counter()

            if cmd is not None:
                cmd = dataclasses.replace(cmd, processing_time=t2 - t1)
                link.send_command(cmd)
            t3 = time.perf_counter()

            overrun = (t3 - t0) > self.period
            self.drain_ms.append((t1 - t0) * 1e3)
            self.process_ms.append((t2 - t1) * 1e3)
            self.total_ms.append((t3 - t0) * 1e3)
            self.counts.append(len(arrived))
            self.overruns += overrun
            self.scans_dropped += n_scans_skipped
            self.camera_dropped += n_camera_skipped
            self.cycles_shedding += bool(n_scans_skipped or n_camera_skipped)

            if self.cycle_table is not None:
                kinds = [m.header.type for m in arrived]            # what arrived; the last two columns say what was skipped of it
                newest = max((m.header.t for m in arrived), default="")
                self.cycle_table.row(cycle, round(t0 - t_begin, 6), newest, len(arrived),
                                     kinds.count(P.MsgType.LIDAR), kinds.count(P.MsgType.CAMERA), kinds.count(P.MsgType.POSE),
                                     round((t1 - t0) * 1e3, 4), round((t2 - t1) * 1e3, 4), round((t3 - t0) * 1e3, 4), int(overrun),
                                     n_scans_skipped, n_camera_skipped)
                if cmd is not None:
                    self.command_table.row(cycle, round(t0 - t_begin, 6), *dataclasses.astuple(cmd))
            cycle += 1
            self.duration = time.perf_counter() - t_begin          # kept up to date: an interrupted run (Ctrl+C) still reports its real length

            if self.realtime:
                next_tick += self.period
                if next_tick < t3:           # late: no catch-up burst
                    next_tick = t3
        self.duration = time.perf_counter() - t_begin

    def stats(self) -> dict:
        """The numbers of the report, for summary.json."""
        n = len(self.total_ms)
        if n == 0:
            return {"cycles": 0}
        out = {"cycles": n, "duration_s": self.duration, "cycles_per_s": n / max(self.duration, 1e-9),
               "target_rate_hz": self.rate_hz, "deadline_ms": self.period * 1e3,
               "messages_per_cycle_mean": float(np.mean(self.counts)), "messages_per_cycle_max": max(self.counts),
               "overruns": self.overruns, "overrun_fraction": self.overruns / n,
               "max_scans_per_cycle": self.shedding.max_scans_per_cycle, "scans_dropped": self.scans_dropped,
               "camera_dropped": self.camera_dropped, "cycles_shedding": self.cycles_shedding}
        for name, v in (("receive", self.drain_ms), ("process", self.process_ms), ("total", self.total_ms)):
            a = np.asarray(v)
            out[f"{name}_ms"] = {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
                                 "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}
        return out

    def report(self) -> str:
        s = self.stats()
        if s["cycles"] == 0:
            return "no cycles ran"

        def row(name: str) -> str:
            m = s[f"{name}_ms"]
            return (f"{name:8s} mean {m['mean']:7.3f}  p50 {m['p50']:7.3f}  p95 {m['p95']:7.3f}  "
                    f"p99 {m['p99']:7.3f}  max {m['max']:7.3f}")

        return "\n".join([
            f"{s['cycles']} cycles in {s['duration_s']:.1f} s ({s['cycles_per_s']:.1f} per second, target {self.rate_hz:g}), "
            f"deadline {s['deadline_ms']:.1f} ms",
            f"messages per cycle: mean {s['messages_per_cycle_mean']:.2f}, max {s['messages_per_cycle_max']}",
            "latency in ms:",
            "  " + row("receive"), "  " + row("process"), "  " + row("total"),
            f"overruns (total > deadline): {s['overruns']} ({100.0 * s['overrun_fraction']:.1f} %)",
            f"load shedding (at most {s['max_scans_per_cycle']} scans per cycle): {s['scans_dropped']} scans and {s['camera_dropped']} camera frames skipped "
            f"in {s['cycles_shedding']} cycles" if s["max_scans_per_cycle"] > 0 else "load shedding: off",
        ])