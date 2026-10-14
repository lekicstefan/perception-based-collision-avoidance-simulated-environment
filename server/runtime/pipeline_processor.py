"""The real processor (step 7.13): perception -> tracking -> risk stage, one command per LiDAR scan.

`PipelineProcessor` replaces the do-nothing `Processor` of cycle.py (same two methods). `step(msgs, calibration)` gets the messages of one cycle,
sorted by capture time, and handles them in that order:
  CAMERA  into the camera buffer (for the refinement of the scans) and as a bearing measurement to the tracker;
  LIDAR   perception, world clusters, tracker step, risk stage; the command of the last scan of the cycle is returned (None without a scan);
  POSE    ignored (the header of every message carries the ego state; the scan header is used).
Everything is built at the first step from the calibration (and again if the calibration changes). `warmup()` has no calibration yet, so the
compiled parts are warmed up by the first scans; nothing is sent before a scan arrives.
Wall time before the stage (perception, tracking) and the age of the scan when the cycle started go into the delay accounting of the stage.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

from server.communication import protocol as P
from server.geometry.calibration import Calibration
from server.perception.pipeline import PerceptionPipeline
from server.risk.cause_log import CauseLog
from server.risk.ego_path import EgoGeometry, EgoState
from server.risk.stage import RiskStage, StageConfig
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
    pose_position_sigma_m: float = 0.0      # what the tracker is told; the same value goes to the stage as ego_sigma_m
    pose_yaw_sigma_rad: float = 0.0
    stage: StageConfig = field(default_factory=StageConfig)


class PipelineProcessor(Processor):
    def __init__(self, config: ProcessorConfig = ProcessorConfig(), log: CauseLog | None = None, clock=time.perf_counter):
        self.config, self.log, self.clock = config, log, clock
        self._calibration_dict = None
        self.calibration = self.pipeline = self.tracker = self.stage = None
        self.last = None                    # StageResult of the last scan
        self.bad_camera_frames = 0

    def _build(self, calibration: dict) -> None:
        cfg = self.config
        cal = Calibration.from_dict(calibration)
        self.calibration = cal
        self.pipeline = PerceptionPipeline(cal)
        noise = MeasurementNoise(sigma_extent_m=cfg.sigma_extent_m, pose_position_sigma_m=cfg.pose_position_sigma_m,
                                 pose_yaw_sigma_rad=cfg.pose_yaw_sigma_rad)
        stage_cfg = replace(cfg.stage, ego_sigma_m=cfg.pose_position_sigma_m)        # one noise for the tracker and for the distance sigma
        self.tracker = Tracker(TrackerConfig(association=AssociationConfig(gate_chi2=cfg.gate_chi2),
                                             filter=FilterConfig(sigma_a=stage_cfg.sigma_a)), noise)
        self.stage = RiskStage(EgoGeometry.from_calibration(cal.ego), self.pipeline.builder.geometry, stage_cfg, self.log, self.clock)
        self._calibration_dict = calibration

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
        h = m.header
        result = self.pipeline.process_message(m.raw)
        img = result.image
        world = clusters_to_world(result.clusters, img, self.pipeline.builder.geometry.origin[:2])
        self.tracker.step(world, img.t)
        outputs = [track_output(tr, self.tracker.config.output) for tr in self.tracker.tracks]
        lead = max(newest - img.t, 0.0) + (self.clock() - t0)
        self.last = self.stage.step(img, result.ground, EgoState(h.speed, h.yaw_rate, h.steering_angle), outputs, lead_s=lead)
        return self.last.command