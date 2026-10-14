"""The COMMAND message of a cycle (step 7.13, spec section 4.3 / protocol.md).

`build_command` is pure: it takes the smoothed caps, the braking request and the risk level and returns `protocol.Command`. Rules:
  - the caps are finite, in [0, ceiling], and comfort <= safety (a comfort cap above the safety cap makes no sense);
  - comfort_decel is the a_comf of the risk level (what Unity uses for the comfortable approach to the cap);
  - the braking request is in [0, a_max]; an emergency asks for a_max;
  - validity = `validity_periods` command periods: when no newer command arrives, Unity applies `fallback_decel` (0 releases);
  - fallback_decel is by default the a_comf of the risk level, so a lost link slows the car gently instead of holding the last cap.
`processing_time` is filled by the cycle runner with the measured value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from server.communication import protocol as P
from server.risk.stopping import StoppingParams


@dataclass(frozen=True)
class CommandConfig:
    validity_periods: float = 2.5           # command periods; 10 Hz LiDAR gives 0.25 s
    command_period_s: float = 0.1
    fallback_decel: float | None = None     # None: the a_comf of the risk level
    max_decel: float = 8.0
    ceiling_mps: float = 40.0

    @property
    def validity_s(self) -> float:
        return self.validity_periods * self.command_period_s


def _cap(x: float, ceiling: float) -> float:
    if x is None or math.isnan(x):
        return 0.0                                                       # no number is not a license to drive
    return min(max(float(x), 0.0), ceiling)


def build_command(command_id: int, data_t: float, comfort_cap: float, safety_cap: float, brake_request: float, emergency: bool,
                  risk_level: int, params: StoppingParams, cfg: CommandConfig = CommandConfig(), processing_time: float = 0.0) -> P.Command:
    safety = _cap(safety_cap, cfg.ceiling_mps)
    comfort = min(_cap(comfort_cap, cfg.ceiling_mps), safety)
    brake = cfg.max_decel if emergency else min(max(float(brake_request or 0.0), 0.0), cfg.max_decel)
    fallback = cfg.fallback_decel if cfg.fallback_decel is not None else params.comfort_decel
    return P.Command(id=int(command_id), data_timestamp=float(data_t), comfort_cap=comfort, safety_cap=safety,
                     comfort_decel=float(params.comfort_decel), brake_request=brake, validity=cfg.validity_s,
                     fallback_decel=min(max(float(fallback), 0.0), cfg.max_decel), processing_time=float(processing_time),
                     emergency=bool(emergency), risk_level=int(risk_level))