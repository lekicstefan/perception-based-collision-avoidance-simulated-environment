"""Hysteresis and smoothing against flicker (step 7.11; section 5.6: "the caps drop immediately and rise gradually, and emergency braking
latches until no hazard requires it").

Every filter is asymmetric: anything that makes the car more cautious acts at once, anything that makes it less cautious is delayed or limited.

  CapSmoother        a speed cap (one for the comfort cap, one for the safety cap): a lower value is taken at once; a higher one is only
                     followed when the raw cap has been above the output for `rise_hold_s` without a break (a track that flickers away
                     and back does not lift the cap), and then at most `rise_mps2` m/s per second. The output never exceeds `ceiling_mps`
                     (a cap without a hazard is infinite; the car's own limit takes its place). A cap that is not a number counts as 0.
  BrakeFilter        the braking request rises at once and falls at most `brake_release_mps3` m/s^2 per second. Full emergency braking
                     latches: once requested it is held until no hazard has required it for `emergency_release_s`, and then the request
                     runs down at the release rate.
  RiskLevelFilter    the risk level goes up at once and comes down one level at a time, after `level_down_hold_s` of a clearly better view
                     (the thresholds of the layer are moved by `level_hysteresis` in the safe direction for the way down, a Schmitt trigger).

Time is the data time of the scan (simulation seconds): the same recording gives the same output. A repeated or older time counts as dt = 0.
Each `update` returns what happened (`limited` / the raw value) for the cause log of step 7.12.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from server.risk.risk_level import RiskAssessment, RiskConfig, RiskLevel, assess
from server.risk.stopping import StoppingParams

_EPS = 1e-9


@dataclass(frozen=True)
class SmoothingConfig:
    comfort_rise_mps2: float = 2.0          # how fast the comfort cap may rise (m/s per s)
    safety_rise_mps2: float = 4.0
    rise_hold_s: float = 0.5                # the raw cap must stay above the output this long before the output rises
    ceiling_mps: float = 40.0               # the highest cap (the car's own limit)
    brake_release_mps3: float = 8.0         # how fast a braking request may fall (m/s^2 per s)
    max_decel_mps2: float = 8.0             # the request while the emergency is latched
    emergency_release_s: float = 0.5        # no emergency requested for this long releases the latch
    level_down_hold_s: float = 2.0          # a better risk level must hold this long before the level comes down (one step)
    level_hysteresis: float = 0.2           # relative move of the thresholds for the way down


@dataclass(frozen=True)
class Smoothed:
    value: float
    raw: float
    limited: bool                           # the output is below the raw cap because of the hold or the rise rate


class CapSmoother:
    def __init__(self, rise_mps2: float, cfg: SmoothingConfig = SmoothingConfig()):
        self.rise = rise_mps2
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.value = None
        self.t = None
        self.above_since = None

    def update(self, t: float, raw: float) -> Smoothed:
        raw = 0.0 if (raw is None or math.isnan(raw)) else min(float(raw), self.cfg.ceiling_mps)
        raw = max(raw, 0.0)
        if self.value is None:
            self.value, self.t, self.above_since = raw, t, None
            return Smoothed(raw, raw, False)
        dt = max(t - self.t, 0.0)
        self.t = max(self.t, t)
        if raw <= self.value:                                                      # lower or equal: at once
            self.value, self.above_since = raw, None
            return Smoothed(raw, raw, False)
        if self.above_since is None:
            self.above_since = t
        if t - self.above_since < self.cfg.rise_hold_s - _EPS:                     # not above for long enough yet
            return Smoothed(self.value, raw, True)
        self.value = min(raw, self.value + self.rise * dt)
        return Smoothed(self.value, raw, self.value < raw)


class BrakeFilter:
    def __init__(self, cfg: SmoothingConfig = SmoothingConfig()):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.brake = 0.0
        self.latched = False
        self.t = None
        self.clear_since = None

    def update(self, t: float, raw_brake: float, raw_emergency: bool) -> tuple:
        """(braking request m/s^2, emergency flag) after the latch and the release rate."""
        dt = 0.0 if self.t is None else max(t - self.t, 0.0)
        self.t = t if self.t is None else max(self.t, t)
        if raw_emergency:
            self.latched, self.clear_since = True, None
        elif self.latched:
            if self.clear_since is None:
                self.clear_since = t
            if t - self.clear_since >= self.cfg.emergency_release_s - _EPS:
                self.latched, self.clear_since = False, None
        target = self.cfg.max_decel_mps2 if self.latched else max(float(raw_brake), 0.0)
        self.brake = max(target, self.brake - self.cfg.brake_release_mps3 * dt)
        return self.brake, self.latched


class RiskLevelFilter:
    def __init__(self, risk_cfg: RiskConfig = RiskConfig(), cfg: SmoothingConfig = SmoothingConfig(), base: StoppingParams = StoppingParams()):
        h = cfg.level_hysteresis
        self.risk_cfg, self.cfg, self.base = risk_cfg, cfg, base
        self.release_cfg = replace(risk_cfg, ratio_high=risk_cfg.ratio_high * (1 + h), ratio_medium=risk_cfg.ratio_medium * (1 + h),
                                   unknown_high=risk_cfg.unknown_high * (1 - h), unknown_medium=risk_cfg.unknown_medium * (1 - h))
        self.reset()

    def reset(self) -> None:
        self.level = None
        self.below_since = None

    def update(self, t: float, sight_distance_m: float, unknown_fraction: float, speed_mps: float) -> tuple:
        """(the assessment with the filtered level and its bundle, the raw level)."""
        raw = assess(sight_distance_m, unknown_fraction, speed_mps, self.risk_cfg, self.base)
        if not self.risk_cfg.enabled or self.level is None:
            self.level, self.below_since = raw.level, None
        elif raw.level > self.level:                                               # more risk: at once
            self.level, self.below_since = raw.level, None
        else:
            rel = assess(sight_distance_m, unknown_fraction, speed_mps, self.release_cfg, self.base)
            if rel.level < self.level:                                             # clearly better than the present level
                if self.below_since is None:
                    self.below_since = t
                if t - self.below_since >= self.cfg.level_down_hold_s - _EPS:
                    self.level = RiskLevel(max(int(rel.level), int(self.level) - 1))   # one step, and the timer starts again
                    self.below_since = None
            else:
                self.below_since = None
        level = RiskLevel(self.level)
        return replace(raw, level=level, bundle=self.risk_cfg.bundles[level]), raw.level