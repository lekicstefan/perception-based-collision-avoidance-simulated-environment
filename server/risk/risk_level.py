"""Risk level: low, medium or high, and the parameter bundle each level selects (step 7.8; section 5.6). A toggleable layer.

Inputs, measured from the range image by `visibility.measure_view` on the corridor of the car:
  sight distance     how far ahead the sensor sees valid free space (to a hazard, an occlusion or a missing return);
  unknown fraction   the share of the corridor cells in front of the first hazard that are invalid or occluded.

Documented function. The sight distance is judged against what the car needs, not in metres: the sight ratio

    q = sight distance / max(d_stop(v), min_reference_m)         d_stop(v) = v t + v^2 / (2 a_ref)       (t and a_ref of the medium bundle)

is the number of stopping distances the car can see. A standing car or a slow one is compared with `min_reference_m`, so that a short view
is not harmless only because the car is slow. The level is the larger of the two criteria

    high    q < ratio_high    or  unknown fraction >= unknown_high
    medium  q < ratio_medium  or  unknown fraction >= unknown_medium
    low     otherwise

The thresholds (defaults below) are initial values, tuned on the tuning sweep points and reported (7.9); a level change needs hysteresis and
smoothing (7.11), which this module leaves to its caller. Every level selects the whole bundle of the table of section 5.6, not only epsilon.
When the layer is off (`RiskConfig.enabled = False`) the level is `fixed_level` whatever the view: the ablation "risk layer off" keeps one bundle.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum

from server.risk.stopping import StoppingParams, stopping_distance


class RiskLevel(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass(frozen=True)
class RiskBundle:
    epsilon: float                  # acceptable risk
    comfort_decel: float            # a_comf (m/s^2)
    assumed_decel: float            # a (m/s^2)
    margin_m: float
    kappa: float                    # covariance inflation
    occluded_speed_mps: float       # occluded approach speed (7.10)
    hazard_decel: float             # a_h (m/s^2)

    def apply(self, base: StoppingParams = StoppingParams()) -> StoppingParams:
        """The stopping parameters of this bundle (delay, jitter and the real maximum deceleration are kept from `base`)."""
        return replace(base, epsilon=self.epsilon, comfort_decel=self.comfort_decel, assumed_decel=self.assumed_decel,
                       margin_m=self.margin_m, kappa=self.kappa, hazard_decel=self.hazard_decel)


BUNDLES = {                                                         # the table of section 5.6
    RiskLevel.LOW: RiskBundle(0.10, 3.0, 6.5, 1.0, 1.0, 1.5, 6.0),
    RiskLevel.MEDIUM: RiskBundle(0.02, 2.5, 6.0, 2.0, 1.5, 2.0, 7.0),
    RiskLevel.HIGH: RiskBundle(0.001, 2.0, 5.0, 3.0, 2.0, 2.5, 8.0),
}


@dataclass(frozen=True)
class RiskConfig:
    enabled: bool = True
    fixed_level: RiskLevel = RiskLevel.MEDIUM       # the level when the layer is off
    ratio_high: float = 1.5                         # q below this: high
    ratio_medium: float = 3.0                       # q below this: medium
    unknown_high: float = 0.35
    unknown_medium: float = 0.15
    min_reference_m: float = 10.0
    bundles: dict = field(default_factory=lambda: dict(BUNDLES))


@dataclass(frozen=True)
class RiskAssessment:
    level: RiskLevel
    sight_ratio: float              # q (inf for no limit)
    sight_distance_m: float
    unknown_fraction: float
    cause: str                      # "off", "sight", "unknown", "both" or "clear" (which criterion set the level)
    bundle: RiskBundle

    def stopping_params(self, base: StoppingParams = StoppingParams()) -> StoppingParams:
        return self.bundle.apply(base)


def reference_distance(speed_mps: float, cfg: RiskConfig = RiskConfig(), base: StoppingParams = StoppingParams()) -> float:
    """What the car needs to see at this speed: its stopping distance with the medium bundle, at least `min_reference_m`."""
    mid = cfg.bundles[RiskLevel.MEDIUM]
    return max(stopping_distance(max(speed_mps, 0.0), base.delay_s, mid.assumed_decel), cfg.min_reference_m)


def assess(sight_distance_m: float, unknown_fraction: float, speed_mps: float, cfg: RiskConfig = RiskConfig(),
           base: StoppingParams = StoppingParams()) -> RiskAssessment:
    q = sight_distance_m / reference_distance(speed_mps, cfg, base)
    if not cfg.enabled:
        return RiskAssessment(cfg.fixed_level, q, sight_distance_m, unknown_fraction, "off", cfg.bundles[cfg.fixed_level])
    by_sight = RiskLevel.HIGH if q < cfg.ratio_high else RiskLevel.MEDIUM if q < cfg.ratio_medium else RiskLevel.LOW
    by_unknown = (RiskLevel.HIGH if unknown_fraction >= cfg.unknown_high
                  else RiskLevel.MEDIUM if unknown_fraction >= cfg.unknown_medium else RiskLevel.LOW)
    level = max(by_sight, by_unknown)
    cause = "clear" if level == RiskLevel.LOW else "both" if by_sight == by_unknown else "sight" if by_sight > by_unknown else "unknown"
    return RiskAssessment(level, q, sight_distance_m, unknown_fraction, cause, cfg.bundles[level])