"""Occlusion-aware virtual hazards from the validity mask (step 7.10; section 5.6, "occlusion-aware capping").

A real car slows down where it cannot see: before a crest, in a blind bend, beside a parked car that hides a pavement. The view field
(visibility.py) says which parts of the ground ahead are hidden; this module puts a virtual hazard at the edge of what is visible, so that the
caps ensure that the car could stop if something appeared there. Nothing here knows what an object is.

  blind corridor   the corridor itself is hidden from the arc length of the first hidden stretch on (`ViewMeasure.limit` is "occluded", "dropout"
                   or "outside"; behind a real hazard nothing is added). The virtual hazard stands still at the edge of the visible part, it
                   can be there at once (tau_first = 0): the car has to be able to stop within what it sees.
  side occlusion   beside the corridor, in a strip of `side_width_m` on each side, cells that are hidden behind an obstacle (a parked car) are
                   where somebody can step out of. For every station the nearest hidden cell lies `gap` m from the corridor edge; the worst
                   case is a pedestrian there who walks towards the corridor at the occluded approach speed v_occ of the risk level, so it is
                   at the corridor edge at tau_first = gap / v_occ. The virtual hazard is at the edge of the corridor, at the first
                   station of a run of at least `min_run_m` hidden stations.

Both go through the same time-of-arrival test as a real hazard (hazards.py): no conflict when the rear of the car has passed the station
`time_margin_s` before the pedestrian can be at the edge (or the station is behind the car); a car that stands still never clears. The result are
`Conflict`s (state "virtual", static, negative track ids, distance from the front bumper at the time of effect) that the stopping model handles
like those of the tracks; at most one per side and one for the corridor, the nearest, because only the nearest sets the cap.
A virtual hazard has no measurement error: its safety margin weight is 0 (the edge of what is visible is already the worst place) and sigma_d is 0.
The cap that follows from a side hazard is floored at the occluded approach speed (`floor_speed_mps`): the stopping model alone would let the
speed go to zero in front of every parked car (v_max(0) = 0) and the car could never pass; a car that creeps by at the speed of the pedestrian
meets whoever steps out at a relative speed of that size. The caller (7.13) applies the floor to the virtual hazards, not to real ones. The blind
corridor has no floor: there the car has to stop within what it sees. The hidden cells directly behind the visible face of a parked car are
counted although the car's own body is there (its depth is not known), so the side hazard stands at the front of the parked car.
Known simplifications: the hidden pedestrian may start at any lateral gap that is hidden (the nearest one is used), a hidden cell behind the
parked car counts although the parked car itself is in the way of nothing, and the hazard stays at the edge (it does not walk on).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from server.risk.ego_path import Corridor, EgoGeometry
from server.risk.hazards import Conflict, HazardConfig
from server.risk.visibility import PointState, ViewConfig, ViewField, ViewMeasure, corridor_cells


@dataclass(frozen=True)
class OcclusionConfig:
    enabled: bool = True
    side_width_m: float = 3.0           # width of the strip beside the corridor that is looked at
    side_step_m: float = 0.25
    side_first_gap_m: float = 0.1       # the nearest lateral sample beyond the corridor edge
    min_run_m: float = 2.0              # a hidden side stretch must be this long (a single hidden cell is noise)
    blind_states: tuple = ("occluded", "dropout", "outside")


@dataclass(frozen=True)
class VirtualHazard:
    kind: str                           # "blind_corridor", "side_left" or "side_right"
    s_m: float                          # arc length of the hazard
    gap_m: float                        # lateral gap of the nearest hidden cell from the corridor edge (0 for the corridor itself)
    tau_first_s: float
    d_path_m: float                     # from the front bumper at the time of effect
    conflict: Conflict
    floor_speed_mps: float = 0.0        # the cap that follows from this hazard is never below this (the occluded approach speed)

    @property
    def cause(self) -> str:
        return f"occlusion:{self.kind}@{self.s_m:.1f}m"


def _conflict(index: int, a: float, b: float, tau: float, d_path: float, t_clear: float, cfg: HazardConfig) -> Conflict:
    return Conflict(track_id=-(index + 1), state="virtual", static=True, tau_first_s=tau, s_conflict_m=a, d_path_m=d_path, segment_m=(a, b),
                    t_clear_s=t_clear, v_along_mps=0.0, v_sigma_mps=0.0, sigma_d_m=0.0, margin_weight=0.0, existence=1.0)


def find_virtual_hazards(view: ViewField, corridor: Corridor, measure: ViewMeasure, ego_speed_mps: float, approach_speed_mps: float,
                         ego_travel_m: float = 0.0, geometry: EgoGeometry = EgoGeometry(), cfg: OcclusionConfig = OcclusionConfig(),
                         hazard_cfg: HazardConfig = HazardConfig(), view_cfg: ViewConfig | None = None) -> list:
    """The virtual hazards of the hidden parts ahead, nearest first. `approach_speed_mps` is the occluded approach speed of the risk level."""
    if not cfg.enabled:
        return []
    vcfg = view_cfg or view.cfg
    tail0 = ego_travel_m - geometry.rear_overhang_m
    out = []

    def clears(b: float, tau: float) -> float | None:
        """t_clear of the segment ending at b, or None when the car is past it before the hazard can be there."""
        t = math.inf if ego_speed_mps < hazard_cfg.min_clear_speed_mps else max(b - tail0, 0.0) / ego_speed_mps
        return None if t + hazard_cfg.time_margin_s < tau else t

    if measure.limit in cfg.blind_states:                                      # the edge of the visible corridor
        a = corridor.front_offset_m + measure.sight_distance_m
        t = clears(a + hazard_cfg.base_radius_m, 0.0)
        if t is not None and a + hazard_cfg.base_radius_m >= tail0:
            d = max(a - corridor.front_offset_m - ego_travel_m, 0.0)
            out.append(VirtualHazard("blind_corridor", a, 0.0, 0.0, d, _conflict(0, a, a + hazard_cfg.base_radius_m, 0.0, d, t, hazard_cfg), 0.0))

    s, lat, pts = corridor_cells(corridor, vcfg)
    k = corridor.k
    ang = k * s
    normal = np.stack([-np.sin(ang), np.cos(ang)], axis=1)
    centre = pts[:, len(lat) // 2]
    gaps = np.arange(cfg.side_first_gap_m, cfg.side_width_m + 1e-9, cfg.side_step_m)
    need = max(int(round(cfg.min_run_m / vcfg.station_step_m)), 1)
    for side, sign in (("side_left", 1.0), ("side_right", -1.0)):
        zone = centre[:, None, :] + sign * (corridor.half_width_m + gaps)[None, :, None] * normal[:, None, :]       # (S, K, 2)
        hidden = view.classify(zone.reshape(-1, 2)).reshape(zone.shape[:2]) == PointState.OCCLUDED
        has = hidden.any(axis=1)
        gap = np.where(has, gaps[np.argmax(hidden, axis=1)], np.inf)
        for i in range(len(s) - need + 1):
            if not has[i:i + need].all():
                continue
            a = float(s[i] - 0.5 * vcfg.station_step_m)                          # the near edge of the station
            b = a + vcfg.station_step_m + hazard_cfg.base_radius_m
            tau = float(gap[i:i + need].min() / max(approach_speed_mps, 1e-6))
            if b < tail0:
                continue
            t = clears(b, tau)
            if t is None:
                continue
            d = max(a - corridor.front_offset_m - ego_travel_m, 0.0)
            out.append(VirtualHazard(side, a, float(gap[i]), tau, d, _conflict(1 if sign > 0 else 2, a, b, tau, d, t, hazard_cfg), approach_speed_mps))
            break                                                                # the nearest of this side only
    return sorted(out, key=lambda h: h.d_path_m)