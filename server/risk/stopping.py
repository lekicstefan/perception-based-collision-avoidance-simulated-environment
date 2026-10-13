"""Stopping model, the two speed caps, the required deceleration and the four response regimes (steps 7.4 and 7.5; formulas of section 5.6).

With the delay t (effective actuation delay + jitter margin; processing delay is never in it, the states are extrapolated to the time
of effect first, step 7.2) and a deceleration a, the stopping distance at speed v is

    d_stop(v) = v t + v^2 / (2 a)                    v_max(a, d) = -a t + sqrt(a^2 t^2 + 2 a d)      (the inverse: d_stop(v_max) = d)

Two caps follow from the effective distance d_eff: the comfort cap v_comf = v_max(a_comf, d_eff) and the safety cap v_safe = v_max(a, d_eff),
with a_comf < a < a_max. The effective distance is

    d_eff = d_path - margin - z_eps kappa sigma_d + v_h,low^2 / (2 a_h)          v_h,low = max(0, v_h - z_eps sigma_v)

(d_path to the conflict point, sigma_d the standard deviation of that distance, kappa the covariance inflation of the risk level, z_eps the
Gaussian quantile of the acceptable risk, a_h the worst-case braking of a hazard that moves ahead along the path; its allowance vanishes
for a static hazard). The deceleration needed to stop within d_eff when braking starts only after t is

    a_req = v^2 / (2 (d_eff - v t))          while d_eff > v t, unbounded otherwise

and v > v_max(x) is the same as a_req > x, so the regimes follow from a_req and the thresholds a_comf < a < a_max:
comfort tracking (a_req <= a_comf), progressive braking (<= a, the car brakes at a_req), hard braking (<= a_max, at a_req) and full emergency
braking (a_req > a_max or d_eff <= v t, at the maximum rate; the aim is then the lowest impact speed, `impact_speed`).
The values here are defaults that equal the examples of the specification; the risk level (step 7.8) replaces the whole bundle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from statistics import NormalDist


class Regime(IntEnum):
    COMFORT = 1         # a_req <= a_comf: hold the comfort cap, slow smoothly as it falls
    PROGRESSIVE = 2     # a_comf < a_req <= a: brake at a_req
    HARD = 3            # a < a_req <= a_max: the safety margin is used, brake at a_req
    EMERGENCY = 4       # a_req > a_max or d_eff <= v t: brake at the maximum rate, minimise the impact speed


@dataclass(frozen=True)
class StoppingParams:
    actuation_delay_s: float = 0.15     # effective actuation delay fitted from the measured stopping table (phase 2, simulator_checks.md): 0.1497 s = 0.10 s delay + half of the 0.1 s jerk ramp (a_max / jerk limit = 8 / 80)
    jitter_margin_s: float = 0.05       # measured gap between the 99th and the 50th percentile latency; PLACEHOLDER until measured (phase 10)
    comfort_decel: float = 3.0          # a_comf (m/s^2)
    assumed_decel: float = 6.0          # a: the deceleration the safety cap assumes, deliberately below the real capability
    max_decel: float = 8.0              # a_max: the real capability of the car (the same fit gives 7.998 m/s^2)
    hazard_decel: float = 6.0           # a_h: worst-case braking of a hazard that moves ahead along the path
    margin_m: float = 0.0
    kappa: float = 1.0                  # covariance inflation
    epsilon: float = 0.10               # acceptable risk: probability that the true distance is shorter than assumed

    @property
    def delay_s(self) -> float:
        """t of the stopping model: actuation delay plus jitter margin."""
        return self.actuation_delay_s + self.jitter_margin_s

    @property
    def z(self) -> float:
        return z_quantile(self.epsilon)


def z_quantile(epsilon: float) -> float:
    """Gaussian quantile for the acceptable risk: P(Z > z) = epsilon (1.28 at 0.10, 2.05 at 0.02, 3.09 at 0.001)."""
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be between 0 and 1")
    return NormalDist().inv_cdf(1.0 - epsilon)


def stopping_distance(v: float, t: float, a: float) -> float:
    """d_stop(v) = v t + v^2 / (2 a)."""
    return v * t + v * v / (2.0 * a)


def v_max(a: float, t: float, d: float) -> float:
    """Highest speed from which the car can stop within d with delay t and deceleration a (0 when d <= 0)."""
    if d <= 0.0:
        return 0.0
    return -a * t + math.sqrt(a * a * t * t + 2.0 * a * d)


def speed_caps(d_eff: float, p: StoppingParams = StoppingParams()) -> tuple:
    """(comfort cap, safety cap) in m/s for the effective distance d_eff."""
    return v_max(p.comfort_decel, p.delay_s, d_eff), v_max(p.assumed_decel, p.delay_s, d_eff)


def lead_speed_low(v_h: float, sigma_v: float, p: StoppingParams = StoppingParams()) -> float:
    """Lower quantile of the speed of a hazard along the path: overestimating a lead's speed is the unsafe direction, so a young or
    noisy track (large sigma_v) gets a small allowance."""
    return max(0.0, v_h - p.z * sigma_v)


def effective_distance(d_path: float, p: StoppingParams = StoppingParams(), sigma_d: float = 0.0, v_h_low: float = 0.0) -> float:
    """d_eff = d_path - margin - z kappa sigma_d + v_h,low^2 / (2 a_h). May be negative (then every cap is 0)."""
    return d_path - p.margin_m - p.z * p.kappa * sigma_d + v_h_low * v_h_low / (2.0 * p.hazard_decel)


def required_deceleration(v: float, d_eff: float, t: float) -> float:
    """a_req = v^2 / (2 (d_eff - v t)) while d_eff > v t, infinity otherwise (0 at standstill)."""
    if v <= 0.0:
        return 0.0
    free = d_eff - v * t
    return math.inf if free <= 0.0 else v * v / (2.0 * free)


def impact_speed(v: float, d_eff: float, t: float, a_max: float) -> float:
    """Speed at the hazard when the car brakes at a_max after the delay t (no steering): v until braking starts, then
    v^2 - 2 a_max (distance left). 0 when it stops in time."""
    left = d_eff - v * t
    if left <= 0.0:
        return v
    return math.sqrt(max(v * v - 2.0 * a_max * left, 0.0))


@dataclass(frozen=True)
class Response:
    regime: Regime
    comfort_cap: float          # m/s
    safety_cap: float           # m/s
    d_eff: float
    a_req: float                # m/s^2, inf when unavoidable
    brake_request: float        # m/s^2 to apply (0 = none)
    emergency: bool
    impact_speed: float         # m/s at the hazard if the car brakes at a_max from now on (0 = stops in time)


def respond(v: float, d_eff: float, p: StoppingParams = StoppingParams()) -> Response:
    """The caps and the regime for speed v and effective distance d_eff."""
    t = p.delay_s
    v_comf, v_safe = speed_caps(d_eff, p)
    a_req = required_deceleration(v, d_eff, t)
    if a_req > p.max_decel:
        regime, brake, emergency = Regime.EMERGENCY, p.max_decel, True
    elif a_req > p.assumed_decel:
        regime, brake, emergency = Regime.HARD, a_req, False
    elif a_req > p.comfort_decel:
        regime, brake, emergency = Regime.PROGRESSIVE, a_req, False
    else:
        regime, brake, emergency = Regime.COMFORT, 0.0, False
    return Response(regime, v_comf, v_safe, d_eff, a_req, brake, emergency, impact_speed(v, d_eff, t, p.max_decel))