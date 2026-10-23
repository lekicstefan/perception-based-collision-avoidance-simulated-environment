from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

from server.communication import protocol as P
from server.geometry.calibration import Calibration
from server.perception.pipeline import PerceptionPipeline
from server.risk.cause_log import CauseLog
from server.risk.ego_path import EgoGeometry, EgoState
from server.risk.stage import RiskStage, StageConfig
from server.risk.telemetry import build_telemetry
from server.runtime.cycle import Processor
from server.tracking.association import AssociationConfig
from server.tracking.kalman import FilterConfig
from server.tracking.measurement import clusters_to_world
from server.tracking.noise import MeasurementNoise
from server.tracking.outputs import track_output
from server.tracking.tracker import Tracker, TrackerConfig


@dataclass(frozen=True)
class ProcessorConfig:
    gate_chi2: float = 18.4                 # tuned in 6.9
    sigma_extent_m: float = 0.4
    sigma_extent_across_m: float = 0.3      # noise of the middle of the visible extent across the line of sight (MeasurementNoise); 0.3 and 0.5 give the same
                                            # result on the parked car (one track id for the box instead of five), 0 is the old behaviour
    pose_position_sigma_m: float = 0.0      # what the tracker is told; the same value goes to the stage as ego_sigma_m
    pose_yaw_sigma_rad: float = 0.0
    stage: StageConfig = field(default_factory=StageConfig)
    telemetry_period_s: float = 0.1         # at most this often (10 Hz); a scan that comes a little early still counts
    telemetry_tolerance_s: float = 0.01


class PipelineProcessor(Processor):
    def __init__(self, config: ProcessorConfig = ProcessorConfig(), log: CauseLog | None = None, clock=time.perf_counter):
        self.config, self.log, self.clock = config, log, clock
        self._calibration_dict = None
        self.calibration = self.pipeline = self.tracker = self.stage = None
        self.last = None                    # StageResult of the last scan
        self.bad_camera_frames = 0
        self.telemetry_sink = None          # a function taking the dict (Link.send_telemetry); None: no telemetry
        self.telemetry_errors = 0           # a failing dashboard channel never disturbs the control
        self._telemetry_t = None

    def _build(self, calibration: dict) -> None:
        self.install(Calibration.from_dict(calibration), calibration)

    def install(self, cal: Calibration, key: dict | None = None) -> None:
        """Builds perception, tracker and stage for a calibration (the closed-loop tests give a Calibration directly)."""
        cfg = self.config
        self.calibration = cal
        self.pipeline = PerceptionPipeline(cal)
        noise = MeasurementNoise(sigma_extent_m=cfg.sigma_extent_m, sigma_extent_across_m=cfg.sigma_extent_across_m,
                                 pose_position_sigma_m=cfg.pose_position_sigma_m, pose_yaw_sigma_rad=cfg.pose_yaw_sigma_rad)
        stage_cfg = replace(cfg.stage, ego_sigma_m=cfg.pose_position_sigma_m)        # one noise for the tracker and for the distance sigma
        self.tracker = Tracker(TrackerConfig(association=AssociationConfig(gate_chi2=cfg.gate_chi2),
                                             filter=FilterConfig(sigma_a=stage_cfg.sigma_a)), noise)
        self.stage = RiskStage(EgoGeometry.from_calibration(cal.ego), self.pipeline.builder.geometry, stage_cfg, self.log, self.clock)
        self._calibration_dict = key

    def step(self, msgs: list, calibration: dict) -> P.Command | None:
        if calibration != self._calibration_dict:
            self._build(calibration)
        t0 = self.clock()
        newest = max((m.header.t for m in msgs), default=0.0)
        cmd = None
        for m in msgs:
            typ = m.header.type
            if typ == P.MsgType.CAMERA:
                self._camera(m)
            elif typ == P.MsgType.LIDAR:
                cmd = self._scan(m, newest, t0)
        return cmd

    def _camera(self, m) -> None:
        try:
            self.pipeline.camera_buffer.add(m.raw)
            h, cam = P.parse_camera(m.raw)
            self.tracker.step_camera(P.camera_to_rgb(cam), h.pose, h.t, self.calibration.camera)
        except P.ProtocolError:
            self.bad_camera_frames += 1

    def _scan(self, m, newest: float, t0: float):
        return self.run_result(self.pipeline.process_message(m.raw), m.header, newest, t0)

    def run_result(self, result, h, newest: float, t0: float):
        """From the perception result of a scan on: tracker, stage, telemetry. `h` has speed, yaw_rate and steering_angle (the scan header);
        `newest` is the newest message time of the cycle, `t0` the clock value at its start."""
        img = result.image
        world = clusters_to_world(result.clusters, img, self.pipeline.builder.geometry.origin[:2])
        self.tracker.step(world, img.t)
        outputs = [track_output(tr, self.tracker.config.output) for tr in self.tracker.tracks]
        pre_s = self.clock() - t0
        lead = max(newest - img.t, 0.0) + pre_s
        self.last = self.stage.step(img, result.ground, EgoState(h.speed, h.yaw_rate, h.steering_angle), outputs, lead_s=lead)
        self._telemetry(self.last, pre_s)
        return self.last.command

    def _telemetry(self, res, pre_s: float) -> None:
        sink, cfg = self.telemetry_sink, self.config
        if sink is None:
            return
        t = res.command.data_timestamp
        if self._telemetry_t is not None and t - self._telemetry_t < cfg.telemetry_period_s - cfg.telemetry_tolerance_s:
            return
        try:
            sink(build_telemetry(res, extra_timing={"perception_tracking_ms": pre_s * 1e3}))
            self._telemetry_t = t
        except Exception:                                    # telemetry is not control: count it, never raise
            self.telemetry_errors += 1