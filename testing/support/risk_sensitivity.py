"""Sensitivity of the two speed caps to the risk parameters (step 7.9). Pure functions, no plotting (the tool plots).

Reference situation: a hazard that stands in the path (v_h = 0) at d_path, or a lead vehicle that drives ahead at v_h with its speed uncertainty
(the only case in which the hazard braking a_h matters); the distance has the standard deviation sigma_d. Every sweep changes ONE parameter and keeps
the rest at the medium bundle ("one at a time"); the tornado compares the effect of the range the three levels span for each parameter; the level
curves give both caps against the distance for every level (and the layer switched off = one fixed bundle).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from server.risk.risk_level import BUNDLES, RiskLevel
from server.risk.stopping import StoppingParams, effective_distance, lead_speed_low, speed_caps, v_max


@dataclass(frozen=True)
class Scene:
    name: str
    d_path_m: float = 30.0
    sigma_d_m: float = 0.5
    v_hazard_mps: float = 0.0
    sigma_v_mps: float = 0.0


STATIC = Scene("hazard in the path")
LEAD = Scene("lead vehicle at 10 m/s", v_hazard_mps=10.0, sigma_v_mps=1.0)

# name -> (StoppingParams field, values of the sweep, label); the bundle parameters first, the others (not in the bundle) last
PARAMETERS = {
    "epsilon": ("epsilon", np.geomspace(0.001, 0.2, 25), "acceptable risk epsilon"),
    "kappa": ("kappa", np.linspace(0.5, 3.0, 26), "covariance inflation kappa"),
    "margin": ("margin_m", np.linspace(0.0, 5.0, 26), "margin (m)"),
    "comfort_decel": ("comfort_decel", np.linspace(1.5, 4.0, 26), "comfortable deceleration a_comf (m/s^2)"),
    "assumed_decel": ("assumed_decel", np.linspace(4.0, 8.0, 26), "assumed deceleration a (m/s^2)"),
    "hazard_decel": ("hazard_decel", np.linspace(4.0, 10.0, 25), "assumed hazard braking a_h (m/s^2)"),
    "delay": ("actuation_delay_s", np.linspace(0.05, 0.40, 36), "actuation delay t (s), not in the bundle"),
}
BUNDLE_FIELDS = {"epsilon": "epsilon", "kappa": "kappa", "margin": "margin_m", "comfort_decel": "comfort_decel",
                 "assumed_decel": "assumed_decel", "hazard_decel": "hazard_decel"}


def medium() -> StoppingParams:
    return BUNDLES[RiskLevel.MEDIUM].apply()


def caps(p: StoppingParams, scene: Scene = STATIC, d_path: float | None = None) -> tuple:
    """(comfort cap, safety cap) of a scene under the parameters p."""
    d = scene.d_path_m if d_path is None else d_path
    v_low = lead_speed_low(scene.v_hazard_mps, scene.sigma_v_mps, p)
    return speed_caps(effective_distance(d, p, scene.sigma_d_m, v_low), p)


def sweep(name: str, scene: Scene = STATIC, base: StoppingParams | None = None) -> list:
    """Rows (value, comfort cap, safety cap) of one parameter, the others at `base` (medium bundle)."""
    field, values, _ = PARAMETERS[name]
    base = base or medium()
    return [(float(v), *caps(replace(base, **{field: float(v)}), scene)) for v in values]


def tornado(scene: Scene = STATIC, base: StoppingParams | None = None) -> list:
    """One row per bundle parameter: (name, low-level value, high-level value, comfort cap change, safety cap change) when ONLY that parameter
    goes from its low-risk to its high-risk value, the rest at medium. Sorted by the larger of the two changes."""
    base = base or medium()
    b_c, b_s = caps(base, scene)
    lo, hi = BUNDLES[RiskLevel.LOW].apply(base), BUNDLES[RiskLevel.HIGH].apply(base)
    rows = []
    for name, field in BUNDLE_FIELDS.items():
        a = caps(replace(base, **{field: getattr(lo, field)}), scene)
        b = caps(replace(base, **{field: getattr(hi, field)}), scene)
        rows.append((name, float(getattr(lo, field)), float(getattr(hi, field)), b[0] - a[0], b[1] - a[1]))
    return sorted(rows, key=lambda r: -max(abs(r[3]), abs(r[4])))


def level_curves(distances, scene: Scene = STATIC) -> dict:
    """level name -> list of (d_path, comfort cap, safety cap)."""
    return {l.name.lower(): [(float(d), *caps(BUNDLES[l].apply(), scene, d)) for d in distances] for l in RiskLevel}


def level_effect(scene: Scene = STATIC) -> dict:
    """Both caps per level at the scene's distance and the relative drop from low to high risk."""
    c = {l.name.lower(): caps(BUNDLES[l].apply(), scene) for l in RiskLevel}
    return {**c, "drop_comfort": 1.0 - c["high"][0] / c["low"][0], "drop_safety": 1.0 - c["high"][1] / c["low"][1]}


def double_count(sigmas=(0.2, 0.5, 1.0), distance: float = 30.0) -> list:
    """The covariance term of the reachable set (7.3) and the sigma_d term of d_eff (7.6) both move the conflict point nearer by kappa z sigma
    for a hazard in the path. Rows (level, sigma, extra distance in m, safety cap counted once, counted twice)."""
    rows = []
    for l in RiskLevel:
        p = BUNDLES[l].apply()
        for s in sigmas:
            extra = p.z * p.kappa * s
            scene = Scene("x", distance, s)
            once = caps(p, scene)[1]
            twice = caps(p, scene, distance - extra)[1]
            rows.append((l.name.lower(), float(s), float(extra), once, twice))
    return rows


def virtual_cap(level: RiskLevel, d_path: float, floor_mps: float | None = None) -> float:
    """Safety cap from a virtual side hazard (no margin, no distance error), floored at the occluded approach speed of the level."""
    b = BUNDLES[level]
    p = replace(b.apply(), margin_m=0.0)
    floor = b.occluded_speed_mps if floor_mps is None else floor_mps
    return max(v_max(p.assumed_decel, p.delay_s, d_path), floor)


def no_conflict_speed(gap_m: float, approach_mps: float, station_m: float, time_margin_s: float = 0.5, rear_m: float = 0.9) -> float:
    """Lowest ego speed at which a hidden pedestrian who walks out at `approach_mps` from `gap_m` beside the corridor is no conflict for a
    station at arc length `station_m` (the car's rear is past it before the pedestrian can be at the edge); inf when no speed does."""
    tau = gap_m / approach_mps
    return float("inf") if tau <= time_margin_s else (station_m + rear_m) / (tau - time_margin_s)