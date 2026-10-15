"""A closed loop without Unity (step 7.15): ray-cast scenes -> the real perception, tracker and risk stage -> commands -> a simple car.

The scene is static objects (boxes) in the world; the car drives along the world x axis. Every `scan_dt_s` the scene is ray-cast from the car's
position (testing.support.synthetic_scene), handed to `PipelineProcessor.run_result` (the real perception, tracking and risk stage, no oracle),
and the command it returns takes effect after `delay_s`. The car model is a stand-in for Unity's command executor:
  emergency: the maximum deceleration; brake_request > 0: that deceleration; speed above min(desired, comfort cap): comfort_decel;
  below it: accelerate at `accel_mps2`. The real executor and the real car are checked end to end in phase 10.
`LoopResult` has one row per scan and the outcome numbers.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

from server.geometry.calibration import Calibration, EgoCalibration
from server.perception.pipeline import PerceptionPipeline
from server.runtime.pipeline_processor import PipelineProcessor, ProcessorConfig
from testing.support.synthetic_camera import make_camera
from testing.support.synthetic_scene import Scene, make_lidar

LIDAR = make_lidar()
CALIB = Calibration("closed_loop", EgoCalibration(2.7, 4.4, 1.8, 1.5, 0.9, 0.8), make_camera(), LIDAR, 100.0)
FRONT_M = 3.5                                   # front bumper ahead of the rear axle
HALF_WIDTH_M = 0.9


@dataclass(frozen=True)
class Box:
    x: float                                    # centre, world
    y: float
    length: float = 4.5
    width: float = 1.8
    height: float = 1.5


@dataclass(frozen=True)
class LoopConfig:
    scan_dt_s: float = 0.1
    sub_dt_s: float = 0.01
    delay_s: float = 0.15                       # from the command to its effect on the car
    desired_mps: float = 12.0
    start_speed_mps: float = 12.0
    accel_mps2: float = 2.0
    max_decel_mps2: float = 8.0
    duration_s: float = 25.0
    stop_after_s: float = 1.0                   # the run ends when the car has stood still this long


@dataclass
class LoopResult:
    rows: list = field(default_factory=list)
    collided: bool = False
    stopped: bool = False

    @property
    def min_gap_m(self) -> float:
        return min((r["gap_m"] for r in self.rows), default=float("inf"))

    @property
    def final(self) -> dict:
        return self.rows[-1]

    @property
    def max_brake(self) -> float:
        return max((r["brake_request"] for r in self.rows), default=0.0)

    @property
    def any_emergency(self) -> bool:
        return any(r["emergency"] for r in self.rows)

    def first_emergency(self):
        return next((r for r in self.rows if r["emergency"]), None)

    def percentile_ms(self, key: str, q: float = 95.0) -> float:
        import numpy as np
        return float(np.percentile([r[key] for r in self.rows], q)) if self.rows else 0.0


def detail(tele) -> str:
    """The strongest response of a scan in words, with the track behind it (for the failure messages of the tests and for dry_run --trace)."""
    rs = tele.decision.responses
    if not rs:
        return "no hazard"
    r = max(rs, key=lambda q: (int(q.regime), q.a_req if math.isfinite(q.a_req) else 1e18))
    c = r.conflict
    out = (f"{r.label} regime={int(r.regime)} in_corridor={getattr(c, 'in_corridor', None)} floor={r.floor_mps} d_path={c.d_path_m:.2f} "
           f"d_eff={r.d_eff_m:.2f} tau={c.tau_first_s:.1f} v_along={c.v_along_mps:.2f}")
    tr = next((p for p in tele.tracks if p.id == c.track_id), None)
    if tr is not None:
        corners = "" if tr.surface is None else " corners=" + str([(round(float(tr.position[0] + q[0]), 1), round(float(tr.position[1] + q[1]), 1))
                                                                   for q in tr.surface[:4]])
        out += (f" | {tr.state} static={tr.static} pos=({tr.position[0]:.1f},{tr.position[1]:.1f}) vel=({tr.velocity[0]:.2f},{tr.velocity[1]:.2f}) "
                f"exist={tr.existence:.2f}{corners}")
    return out


def gap_to(boxes: list, x: float) -> float:
    """Distance from the front bumper to the nearest box rear face ahead that overlaps the car's width (inf if none)."""
    best = float("inf")
    for b in boxes:
        if abs(b.y) - b.width / 2 < HALF_WIDTH_M:
            best = min(best, b.x - b.length / 2 - (x + FRONT_M))
    return best


def run_loop(boxes: list, cfg: LoopConfig = LoopConfig(), processor_cfg: ProcessorConfig = ProcessorConfig()) -> LoopResult:
    proc = PipelineProcessor(processor_cfg)
    proc.install(CALIB)
    res = LoopResult()
    x, v, t = 0.0, cfg.start_speed_mps, 0.0
    pending, active = [], None                   # (time of effect, command)
    still = 0.0
    for k in range(int(cfg.duration_s / cfg.scan_dt_s)):
        scene = Scene()
        for b in boxes:
            scene.add_box(b.x - x, b.y, b.length, b.width, b.height)
        img, _, _ = scene.render(LIDAR)
        img = type(img)(t, k, (x, 0.0, 0.0, 0.0, 0.0, 0.0), v, 0.0, 0.0, img.range_m, img.state)
        w0 = time.perf_counter()
        result = proc.pipeline.process(img)
        w1 = time.perf_counter()
        cmd = proc.run_result(result, SimpleNamespace(speed=v, yaw_rate=0.0, steering_angle=0.0), t, w1)
        w2 = time.perf_counter()
        pending.append((t + cfg.delay_s, cmd))
        tele = proc.last
        res.rows.append({"t": t, "x": x, "v": v, "gap_m": gap_to(boxes, x), "comfort_cap": cmd.comfort_cap, "safety_cap": cmd.safety_cap,
                         "brake_request": cmd.brake_request, "emergency": cmd.emergency, "risk_level": cmd.risk_level,
                         "n_tracks": tele.n_tracks, "perception_ms": (w1 - w0) * 1e3, "tracking_stage_ms": (w2 - w1) * 1e3,
                         "total_ms": (w2 - w0) * 1e3, "stage_ms": tele.timings_ms["total"],
                         "regime": int(tele.decision.regime), "raw_safety_cap": tele.decision.safety_cap,
                         "brake_cause": tele.record["brake_cause"]["label"], "brake_d_path_m": tele.record["brake_cause"]["d_path_m"],
                         "brake_d_eff_m": tele.record["brake_cause"]["d_eff_m"], "brake_tau_s": tele.record["brake_cause"]["tau_first_s"],
                         "comfort_cause": tele.record["comfort_cause"]["label"], "detail": detail(tele)})
        for _ in range(round(cfg.scan_dt_s / cfg.sub_dt_s)):
            while pending and pending[0][0] <= t + 1e-9:
                active = pending.pop(0)[1]
            a = _accel(v, active, cfg)
            v = max(v + a * cfg.sub_dt_s, 0.0)
            x += v * cfg.sub_dt_s
            t += cfg.sub_dt_s
            if gap_to(boxes, x) <= 0.0:
                res.collided = True
        still = still + cfg.scan_dt_s if v < 0.05 else 0.0
        if res.collided or still >= cfg.stop_after_s:
            break
    res.stopped = v < 0.05
    res.rows.append({**res.rows[-1], "t": t, "x": x, "v": v, "gap_m": gap_to(boxes, x)})
    return res


def _accel(v: float, cmd, cfg: LoopConfig) -> float:
    if cmd is None:
        return 0.0
    if cmd.emergency:
        return -cfg.max_decel_mps2
    if cmd.brake_request > 0.0:
        return -min(cmd.brake_request, cfg.max_decel_mps2)
    target = min(cfg.desired_mps, cmd.safety_cap, cmd.comfort_cap)
    if v > target + 0.05:
        return -cmd.comfort_decel
    if v < target - 0.05:
        return cfg.accel_mps2
    return 0.0