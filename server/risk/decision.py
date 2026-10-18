"""The decision of one cycle: what every hazard asks for, the binding causes and the raw command values (steps 7.5, 7.10 and 7.12; section 5.6).

Every hazard (the conflicts of the tracks of 7.3 and the virtual hazards of 7.10) is answered by the stopping model on its own:

    margin       the margin of the risk level times the margin weight of the hazard (tentative track 0.5, virtual hazard 0)
    d_eff        effective_distance(d_path, ..., sigma_d, lower quantile of the lead's speed)
    caps         comfort and safety cap from d_eff; a virtual side hazard has a floor (the occluded approach speed): its caps are never below it
    floor        a hazard that is only a precaution (a person beside the road that could step in, a hidden strip beside a parked car) does not stop the car:
                 its caps are never below a floor, the PASSING SPEED (DecisionConfig.passing_speed_mps; the live system uses 40 km/h: the system
                 reacts, it does not drive for the driver). With a passing speed of 0 the floor is the occluded approach speed of the risk level (the
                 earlier behaviour). The floor is a speed RELATIVE to the hazard: a precaution-only hazard that moves ahead along the path (a person walking along the
                 pavement) is passed at its own speed (lower quantile) plus the floor, but only for an established track (side_floor_relative_min_age_s):
                 the velocity of a young track, such as a sliver of a standing box, is a systematic error that its covariance does not show, and it
                 must not lift the floor
    regime       from the required deceleration a_req (stopping.respond); for a hazard with a floor the car has to slow down to the floor, not to a
                 stop: a_req = (v^2 - floor^2) / (2 (d_path - v t)) and nothing at all while v <= floor. The NOMINAL distance d_path is used here, not
                 d_eff: the caps carry margin and covariance (they start the slow-down early), and a_req only asks for more when the car physically
                 cannot get down to the floor before the conflict point. (With d_eff a car that is slightly above the floor and nearer than margin plus
                 covariance, about 3.5 m, would be told to brake at the maximum.) A hazard with a floor (a precaution or a hidden strip) asks for no
                 braking once the car is at its conflict point (no distance is left to slow down in, the floored caps still hold the speed; without this a
                 car creeping beside a parked car and a hair above the floor was told to brake at the maximum, again and again)

The command takes the lowest cap over all hazards, and the strongest regime. Each of these has a CAUSE: the hazard that set it, as a `Cause`
(a track id, or a virtual occlusion hazard with its place), with the numbers the evaluation needs to join it with the ground truth (section 5.10:
realized-hazard, precautionary or unjustified): state and existence of the track, whether it is flagged static, the distance, the first time the
reachable set can be in the corridor. Without any hazard, or when the hazard asks for more than the ceiling, the cause is `NO_CAUSE`.
Smoothing (7.11) and the emergency latch come after this; the cause log (cause_log.py) says when the output is held by them.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

from server.risk.hazards import Conflict
from server.risk.occlusion import VirtualHazard
from server.risk.stopping import Regime, StoppingParams, effective_distance, lead_speed_low, respond


@dataclass(frozen=True)
class DecisionConfig:
    ceiling_mps: float = 40.0           # the cap when nothing limits the speed (the car's own limit)
    emergency_min_speed_mps: float = 1.0    # below this speed "no distance left" is not an emergency: the car brakes at the maximum (regime HARD)
                                            # but no emergency is flagged or latched (a car creeping to a stop inside its margin is not an event)
    side_floor_enabled: bool = True         # a conflict that exists only because the object COULD reach the corridor (its nominal motion stays out of it:
                                            # a parked car, a person on the pavement) gets the occluded approach speed as a floor, like a hidden side strip,
                                            # so the car passes at that speed instead of stopping beside it
    passing_speed_mps: float = 0.0          # the speed at which the car passes a hazard that is only a precaution (a person beside the road, a hidden side strip
                                            # beside a parked car): the caps of such a hazard are never below it. 0: the occluded approach speed of the risk level
                                            # (the earlier behaviour); the live system uses 40 km/h, see run_processor --passing-speed-kmh. A hidden corridor ahead
                                            # (blind corridor) and an object in the lane keep their stop distance
    side_floor_relative: bool = True        # the floor is raised by the hazard's own speed along the path (lower quantile) ...
    side_floor_relative_min_age_s: float = 3.0      # ... but only for a track at least this old (a sliver of a standing box lives a few scans)
    emergency_needs_physical: bool = True   # an emergency is flagged only when the car cannot stop before the hazard at the maximum deceleration even
                                            # without margin and covariance; when only the margin is used up it brakes at the maximum as HARD


@dataclass(frozen=True)
class Cause:
    kind: str = "none"                  # "track", "virtual" or "none"
    source_id: int | None = None        # track id (>= 0) or the negative id of the virtual hazard
    label: str = "none"                 # "track:12" or "occlusion:side_left@28.5m"
    state: str = ""                     # track state ("confirmed", "tentative", "coasting") or "virtual"
    static: bool = False
    d_path_m: float = math.inf
    d_eff_m: float = math.inf
    cap_mps: float = math.inf           # the cap this hazard asks for
    a_req_mps2: float = 0.0
    regime: int = int(Regime.COMFORT)
    tau_first_s: float = 0.0
    existence: float = 1.0
    margin_weight: float = 1.0
    floor_mps: float = 0.0              # the floor of this hazard (0: none), including the part that comes from the hazard's own speed
    v_along_mps: float = 0.0            # the hazard's speed along the path at the conflict point, and its standard deviation
    v_sigma_mps: float = 0.0
    age_s: float = math.inf             # the age of the track (null in the log: unknown)

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in d.items()}


NO_CAUSE = Cause()


@dataclass(frozen=True)
class HazardResponse:
    conflict: Conflict
    kind: str
    label: str
    floor_mps: float
    d_eff_m: float
    comfort_cap: float
    safety_cap: float
    a_req: float
    regime: Regime
    brake_request: float
    emergency: bool

    def cause(self, cap: float = math.inf) -> Cause:
        c = self.conflict
        return Cause(self.kind, c.track_id, self.label, c.state, c.static, c.d_path_m, self.d_eff_m, cap, self.a_req, int(self.regime),
                     c.tau_first_s, c.existence, c.margin_weight, self.floor_mps, c.v_along_mps, c.v_sigma_mps, c.age_s)


@dataclass(frozen=True)
class Decision:
    t: float                            # data time of the scan
    speed_mps: float
    comfort_cap: float                  # raw: the lowest over the hazards, at most the ceiling
    safety_cap: float
    comfort_cause: Cause
    safety_cause: Cause
    regime: Regime
    brake_request: float                # m/s^2
    emergency: bool
    brake_cause: Cause
    risk_level: int
    n_hazards: int
    n_virtual: int
    responses: tuple = ()


def required_deceleration_to(v: float, target: float, d_eff: float, t: float) -> float:
    """Deceleration that takes the speed from v to `target` within d_eff when braking only starts after t (inf when there is no distance)."""
    if v <= target:
        return 0.0
    free = d_eff - v * t
    return math.inf if free <= 0.0 else (v * v - target * target) / (2.0 * free)


def classify(a_req: float, p: StoppingParams) -> tuple:
    """(regime, braking request, emergency) from a required deceleration (the thresholds of stopping.respond)."""
    if a_req > p.max_decel:
        return Regime.EMERGENCY, p.max_decel, True
    if a_req > p.assumed_decel:
        return Regime.HARD, a_req, False
    if a_req > p.comfort_decel:
        return Regime.PROGRESSIVE, a_req, False
    return Regime.COMFORT, 0.0, False


def respond_to_hazard(conflict: Conflict, speed_mps: float, params: StoppingParams, floor_mps: float = 0.0, label: str | None = None,
                      kind: str = "track", relative_floor_min_age_s: float | None = None) -> HazardResponse:
    p = replace(params, margin_m=params.margin_m * conflict.margin_weight)
    v_low = lead_speed_low(conflict.v_along_mps, conflict.v_sigma_mps, p)
    d_eff = effective_distance(conflict.d_path_m, p, conflict.sigma_d_m, v_low)
    r = respond(speed_mps, d_eff, p)
    label = label if label is not None else f"track:{conflict.track_id}"
    if floor_mps <= 0.0:
        return HazardResponse(conflict, kind, label, 0.0, d_eff, r.comfort_cap, r.safety_cap, r.a_req, r.regime, r.brake_request, r.emergency)
    if relative_floor_min_age_s is not None and conflict.age_s >= relative_floor_min_age_s:
        floor_mps += v_low      # the floor is a speed RELATIVE to the hazard: one that moves ahead along the path is passed at its own speed plus the floor
    a_req = required_deceleration_to(speed_mps, floor_mps, conflict.d_path_m, p.delay_s)      # nominal distance: the margin is in the caps (see above)
    if conflict.d_path_m - speed_mps * p.delay_s <= 0.0:
        a_req = 0.0             # a hazard with a floor (a precaution, a hidden strip) and the car is already at its conflict point: nothing is left to brake for,
                                # the floored caps hold the speed
    regime, brake, emergency = classify(a_req, p)
    return HazardResponse(conflict, kind, label, floor_mps, d_eff, max(r.comfort_cap, floor_mps), max(r.safety_cap, floor_mps), a_req, regime,
                          brake, emergency)


def physically_unavoidable(r: HazardResponse, speed_mps: float, params: StoppingParams) -> bool:
    """True when even the nominal distance (no margin, no covariance allowance) is too short to take the speed down to the hazard's floor speed
    at the maximum deceleration after the delay."""
    free = r.conflict.d_path_m - speed_mps * params.delay_s
    if free <= 0.0:
        return speed_mps > r.floor_mps
    if speed_mps <= r.floor_mps:
        return False
    return (speed_mps ** 2 - r.floor_mps ** 2) / (2.0 * free) > params.max_decel


def _soften(r: HazardResponse, speed_mps: float, params: StoppingParams, cfg: DecisionConfig) -> HazardResponse:
    """An "emergency" that only uses up the margin (or happens at a creeping speed) brakes at the maximum but is the HARD regime, not an emergency."""
    if not r.emergency:
        return r
    if speed_mps < cfg.emergency_min_speed_mps or (cfg.emergency_needs_physical and not physically_unavoidable(r, speed_mps, params)):
        return replace(r, emergency=False, regime=Regime.HARD)
    return r


def decide(t: float, speed_mps: float, conflicts: list, virtuals: list, params: StoppingParams, risk_level: int = 1,
           cfg: DecisionConfig = DecisionConfig(), side_floor_mps: float = 0.0) -> Decision:
    """The raw decision of a cycle from the conflicts of the tracks and the virtual hazards (nearest first or not: the order does not matter)."""
    track_floor = cfg.passing_speed_mps if cfg.passing_speed_mps > 0.0 else side_floor_mps
    responses = [respond_to_hazard(c, speed_mps, params, track_floor if (cfg.side_floor_enabled and not c.in_corridor) else 0.0,
                                   relative_floor_min_age_s=cfg.side_floor_relative_min_age_s if cfg.side_floor_relative else None) for c in conflicts]
    responses += [respond_to_hazard(v.conflict, speed_mps, params,
                                    cfg.passing_speed_mps if (cfg.passing_speed_mps > 0.0 and v.floor_speed_mps > 0.0) else v.floor_speed_mps,
                                    v.cause, "virtual") for v in virtuals]
    responses = [_soften(r, speed_mps, params, cfg) for r in responses]
    if not responses:
        return Decision(t, speed_mps, cfg.ceiling_mps, cfg.ceiling_mps, NO_CAUSE, NO_CAUSE, Regime.COMFORT, 0.0, False, NO_CAUSE, int(risk_level),
                        0, 0)
    comfort = min(responses, key=lambda r: (r.comfort_cap, r.conflict.d_path_m))
    safety = min(responses, key=lambda r: (r.safety_cap, r.conflict.d_path_m))
    strongest = max(responses, key=lambda r: (int(r.regime), r.a_req if math.isfinite(r.a_req) else 1e18, -r.conflict.d_path_m))
    comfort_cap, safety_cap = min(comfort.comfort_cap, cfg.ceiling_mps), min(safety.safety_cap, cfg.ceiling_mps)
    return Decision(
        t=t, speed_mps=speed_mps, comfort_cap=comfort_cap, safety_cap=safety_cap,
        comfort_cause=comfort.cause(comfort_cap) if comfort.comfort_cap < cfg.ceiling_mps else NO_CAUSE,
        safety_cause=safety.cause(safety_cap) if safety.safety_cap < cfg.ceiling_mps else NO_CAUSE,
        regime=strongest.regime, brake_request=max(r.brake_request for r in responses), emergency=any(r.emergency for r in responses),
        brake_cause=strongest.cause(strongest.comfort_cap) if strongest.regime > Regime.COMFORT else NO_CAUSE,
        risk_level=int(risk_level), n_hazards=len(conflicts), n_virtual=len(virtuals), responses=tuple(responses))