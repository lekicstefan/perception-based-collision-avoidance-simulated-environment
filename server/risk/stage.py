"""The risk stage of one LiDAR scan (step 7.13): from the tracks and the scan to the command.

Order, per scan (data time t, ego state from the scan header):
  corridor -> view (ViewField, measure_view) -> risk level (filtered) and its bundle -> time of effect (delay accounting) ->
  predicted tracks (all tracks incl. tentative; the hazard layer weighs them) -> conflicts (7.3) and virtual hazards (7.10) ->
  [collision probability (7.7), informative] -> decision (7.12) -> smoothing (7.11) -> command (command.py) -> cause log.
The latency sample of the delay accounting is the time the data was already old when this stage started (`lead_s`: the newest message time
minus the scan time, plus the wall time of perception and tracking) plus the wall time of the stage itself.
`timings_ms` of the result has the wall time of each part, for the 33 ms budget.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from server.risk.cause_log import CauseLog
from server.risk.collision import CollisionConfig, CollisionResult, collision_probability
from server.risk.command import CommandConfig, build_command
from server.risk.decision import Decision, DecisionConfig, decide
from server.risk.ego_path import EgoGeometry, EgoPathConfig, EgoState, make_corridor
from server.risk.hazards import HazardConfig, find_conflicts
from server.risk.occlusion import OcclusionConfig, find_virtual_hazards
from server.risk.risk_level import RiskAssessment, RiskConfig
from server.risk.smoothing import BrakeFilter, CapSmoother, RiskLevelFilter, SmoothingConfig
from server.risk.stopping import StoppingParams
from server.risk.timing import DelayAccounting, DelayConfig, ego_travel, predict_track
from server.risk.visibility import ViewConfig, ViewField, measure_view
from server.tracking.outputs import track_output


@dataclass(frozen=True)
class StageConfig:
    sigma_a: float = 1.4                    # process noise of the tracker's filter (must be the tracker's)
    ego_sigma_m: float = 0.0                # pose noise of the car's own position; the same value goes into the tracker's MeasurementNoise
    stopping: StoppingParams = field(default_factory=StoppingParams)
    ego_path: EgoPathConfig = field(default_factory=EgoPathConfig)
    view: ViewConfig = field(default_factory=ViewConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    hazard: HazardConfig = field(default_factory=HazardConfig)
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    delay: DelayConfig = field(default_factory=DelayConfig)
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    with_collision: bool = True


@dataclass
class StageResult:
    command: object
    decision: Decision
    assessment: RiskAssessment
    raw_level: int
    collision: CollisionResult | None
    n_tracks: int
    sigma_scale: float
    t_effect: float
    latency_s: float                        # smoothed latency after this cycle
    timings_ms: dict
    record: dict
    tracks: list = field(default_factory=list)          # PredictedTrack at the time of effect (vehicle frame of the data time)
    virtuals: list = field(default_factory=list)
    measure: object = None                              # ViewMeasure
    ego: object = None                                  # EgoState


class RiskStage:
    def __init__(self, geometry: EgoGeometry, lidar_geometry, cfg: StageConfig = StageConfig(), log: CauseLog | None = None,
                 clock=time.perf_counter):
        self.geometry, self.lidar_geometry, self.cfg, self.clock = geometry, lidar_geometry, cfg, clock
        self.log = log if log is not None else CauseLog()
        sm = cfg.smoothing
        self.level_filter = RiskLevelFilter(cfg.risk, sm, cfg.stopping)
        self.comfort = CapSmoother(sm.comfort_rise_mps2, sm)
        self.safety = CapSmoother(sm.safety_rise_mps2, sm)
        self.brake = BrakeFilter(sm)
        self.delay = DelayAccounting(cfg.delay)
        self.rng = np.random.default_rng(cfg.collision.seed)
        self.next_id = 1

    def step(self, img, ground, ego: EgoState, outputs: list, lead_s: float = 0.0, processing_s: float = 0.0) -> StageResult:
        """`img`: RangeImage of the scan (t, pose); `ground`: its ground mask; `outputs`: TrackOutput of every track to consider."""
        cfg, clk = self.cfg, self.clock
        t0 = clk()
        t, pose, speed = float(img.t), img.pose, float(ego.speed_mps)
        tm = {}

        corridor = make_corridor(ego, self.geometry, cfg.ego_path)
        view = ViewField(img, ground, self.lidar_geometry, cfg.view)
        measure = measure_view(view, corridor, cfg.view)
        tm["view"] = (clk() - t0) * 1e3
        a = clk()
        assessment, raw_level = self.level_filter.update(t, measure.sight_distance_m, measure.unknown_fraction, speed)
        params = assessment.stopping_params(cfg.stopping)

        t_effect = self.delay.time_of_effect(t)
        travel = ego_travel(speed, self.delay.horizon_s)
        tracks = [predict_track(o, t_effect, pose, cfg.sigma_a) for o in outputs]
        sigma_scale = params.kappa * params.z
        conflicts = find_conflicts(tracks, corridor, speed, travel, self.geometry, cfg.hazard, sigma_scale, cfg.ego_sigma_m)
        virtuals = find_virtual_hazards(view, corridor, measure, speed, assessment.bundle.occluded_speed_mps, travel, self.geometry,
                                        cfg.occlusion, cfg.hazard, cfg.view)
        tm["hazards"] = (clk() - a) * 1e3
        a = clk()
        coll = None
        if cfg.with_collision and tracks:
            coll = collision_probability(tracks, corridor, speed, travel, self.geometry, cfg.collision, cfg.hazard, sigma_scale, self.rng, clk)
        tm["collision"] = (clk() - a) * 1e3

        a = clk()
        d = decide(t, speed, conflicts, virtuals, params, int(assessment.level), cfg.decision)
        comfort, safety = self.comfort.update(t, d.comfort_cap), self.safety.update(t, d.safety_cap)
        brake, latched = self.brake.update(t, d.brake_request, d.emergency)
        emergency = bool(d.emergency or latched)
        cid = self.next_id
        self.next_id += 1
        cmd = build_command(cid, t, comfort.value, safety.value, brake, emergency, int(assessment.level), params, cfg.command, processing_s)
        rec = self.log.record(d, comfort_cap=cmd.comfort_cap, safety_cap=cmd.safety_cap, comfort_held=comfort.limited, safety_held=safety.limited,
                              brake_request=cmd.brake_request, emergency=cmd.emergency, latched=bool(latched), command_id=cid)
        tm["decision"] = (clk() - a) * 1e3
        tm["total"] = (clk() - t0) * 1e3
        self.delay.update(lead_s + tm["total"] * 1e-3)
        return StageResult(cmd, d, assessment, int(raw_level), coll, len(tracks), sigma_scale, t_effect, self.delay.latency_s, tm, rec,
                           tracks, virtuals, measure, ego)