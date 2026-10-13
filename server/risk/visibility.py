"""What the sensor sees of the ground ahead: the view field and the two measurements of the risk level (step 7.8; section 5.6).

The range image is turned into a per-column description (a column is one azimuth). A point of the ground plane (x, y in the vehicle frame) is
looked up in the column of its azimuth, at its horizontal distance rho from the sensor, and gets one of the states of `PointState`:

  FREE      the line of sight reaches it and the rays that pass through the height of the objects (0 .. obstacle_max_height_m) there have
            returns;
  BLOCKED   an obstacle (a return that is not ground and lower than obstacle_max_height_m) is at rho +- occlusion_tol_m;
  OCCLUDED  it is behind an obstacle of its column, or behind the end of the visible ground (a crest or an edge: the beams that should
            have met ground there passed over it);
  DROPOUT   more than `dropout_fraction` of the rays through its height have no return (NO_RETURN is unknown, never free space), or no ray
            passes through it;
  OUTSIDE   outside the horizontal field of view;
  RANGE     beyond the maximum range of the sensor;
  NEAR      closer than the steepest beam reaches the ground, which is not looked at (the car has just driven over it).
FREE and the "unknown" states DROPOUT, OUTSIDE, OCCLUDED and RANGE are what the risk level counts; BLOCKED is a hazard (the track conflict
test of 7.3 deals with it) and NEAR is skipped. The cells follow the arc of the corridor up to 0.9 pi / |k| (a bend that turns the path round has no "ahead" beyond it). The same field is used by step 7.10 (virtual hazards at occlusion edges).

The two measurements, on the corridor of the car (stations every `station_step_m` along its arc, `lateral_step_m` across it):
  sight distance    the distance from the front bumper to the first station that is blocked by an obstacle, or whose unknown share is at
                    least `station_unknown_fraction` for at least `min_run_m` of stations; `eval_range_m` (or the end of the corridor)
                    when there is none. `limit` says why it ends: "hazard", "occluded", "dropout", "outside", "range" or "clear";
  unknown fraction  the share of the corridor cells that are unknown, counted over the stations in front of the first hazard (what is
                    behind a hazard is the hazard's business) up to eval_range_m; 0 when nothing is evaluated.
Approximations: flat-ground model for the end of the visible ground (a slope that is not level with the pose can fool it), pitch and roll are
not used, the azimuth of a column is taken at the beam nearest to the horizon.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from server.perception.range_image import CellState, LidarGeometry, RangeImage
from server.risk.ego_path import Corridor


class PointState(IntEnum):
    FREE = 0
    DROPOUT = 1
    OUTSIDE = 2
    OCCLUDED = 3
    BLOCKED = 4
    NEAR = 5
    RANGE = 6


UNKNOWN_STATES = (PointState.DROPOUT, PointState.OUTSIDE, PointState.OCCLUDED, PointState.RANGE)


@dataclass(frozen=True)
class ViewConfig:
    obstacle_max_height_m: float = 2.5      # returns higher than this (signs, bridges, branches) do not block the road
    occlusion_tol_m: float = 0.5            # a point this close to the first obstacle of its column is "blocked", behind it "occluded"
    dropout_fraction: float = 0.5           # share of NO_RETURN among the rays through the height of the objects: more than it is unknown
    station_step_m: float = 1.0
    lateral_step_m: float = 0.2
    eval_range_m: float = 60.0
    min_run_m: float = 2.0                  # an unknown stretch must be this long to end the sight distance (a single dropout does not)
    station_unknown_fraction: float = 0.5   # share of the cells of a station that must be unknown for the station to count as unknown


class ViewField:
    """The lookup of a scan: classify(points) -> PointState per point."""

    def __init__(self, img: RangeImage, ground: np.ndarray, geometry: LidarGeometry, cfg: ViewConfig = ViewConfig()):
        self.cfg = cfg
        self.max_range = float(geometry.max_range_m)
        d = geometry.dirs_vehicle
        o = np.asarray(geometry.origin, dtype=float)
        self.o_xy, self.o_z = o[:2], float(o[2])
        hz = np.hypot(d[..., 0], d[..., 1])
        self.slope = d[..., 2] / np.maximum(hz, 1e-9)                              # (rows, cols): height gained per metre of horizontal distance
        row0 = int(np.argmin(np.abs(self.slope[:, self.slope.shape[1] // 2])))   # the beam nearest to the horizon gives the azimuths
        az = np.arctan2(d[row0, :, 1], d[row0, :, 0])
        self._az_asc = az[::-1] if az[0] > az[-1] else az
        self._flip = bool(az[0] > az[-1])
        self._half_step = float(np.abs(np.diff(az)).mean()) / 2.0 if len(az) > 1 else math.pi
        self.state = img.state
        rho = img.range_m.astype(np.float64) * hz
        z = self.o_z + img.range_m.astype(np.float64) * d[..., 2]
        obstacle = img.valid & ~np.asarray(ground, dtype=bool) & (z <= cfg.obstacle_max_height_m)
        self.rho_obs = np.min(np.where(obstacle, rho, np.inf), axis=0)              # (cols,) first obstacle of every column
        self.rho_ground_limit = self._ground_limit(img, np.asarray(ground, dtype=bool), rho)
        down = self.slope[:, self.slope.shape[1] // 2]
        self.near_m = self.o_z / max(-float(down.min()), 1e-9) if down.min() < 0 else 0.0

    def _ground_limit(self, img: RangeImage, ground: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """Per column: the distance beyond which the ground is no longer seen although the next beam should have met it (crest, edge).

        Along a column the beams are followed from the steepest upwards as long as they return ground (a dropout does not break the chain).
        The last two ground returns give the local slope g of the ground; the next beam (slope m < g) would meet the extended ground at
        rho_exp. When it did not (it passed over: no return within range), the ground beyond rho_exp is not seen. A beam with m >= g never
        meets the ground (the horizon of a flat road), and a rho_exp beyond the maximum range cannot be judged: no limit in both cases."""
        rows, cols = img.state.shape
        mid = self.slope[:, cols // 2]
        order = np.argsort(mid)                                                      # steepest beam first
        g = ground[order]
        nr = img.state[order] == CellState.NO_RETURN
        chain = np.logical_and.accumulate(g | nr, axis=0)
        rho_o = rho[order]
        z_o = np.where(g, self.o_z + self.slope[order] * rho_o, np.nan)             # height of the ground returns
        limit = np.full(cols, np.inf)
        for c in range(cols):
            idx = np.flatnonzero(chain[:, c] & g[:, c])
            if len(idx) == 0:
                continue
            k = int(idx[-1])
            if k + 1 >= rows:
                continue
            slope_g = 0.0
            if len(idx) >= 2 and rho_o[k, c] - rho_o[idx[-2], c] > 1e-6:
                slope_g = float(np.clip((z_o[k, c] - z_o[idx[-2], c]) / (rho_o[k, c] - rho_o[idx[-2], c]), -0.5, 0.5))
            m = mid[order][k + 1]
            if m >= slope_g:
                continue
            exp = (z_o[k, c] - slope_g * rho_o[k, c] - self.o_z) / (m - slope_g)
            if rho_o[k, c] < exp <= self.max_range:
                limit[c] = exp
        return limit

    def classify(self, points_xy: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        p = np.atleast_2d(np.asarray(points_xy, dtype=float)) - self.o_xy
        rho = np.hypot(p[:, 0], p[:, 1])
        az = np.arctan2(p[:, 1], p[:, 0])
        n = len(az)
        idx = np.clip(np.searchsorted(self._az_asc, az), 1, len(self._az_asc) - 1)
        left_closer = np.abs(az - self._az_asc[idx - 1]) <= np.abs(self._az_asc[idx] - az)
        col_asc = np.where(left_closer, idx - 1, idx)
        col = (len(self._az_asc) - 1 - col_asc) if self._flip else col_asc
        outside = (az < self._az_asc[0] - self._half_step) | (az > self._az_asc[-1] + self._half_step)
        out = np.full(n, PointState.FREE, dtype=np.int8)
        # unknown because of dropouts: rays through the height of the objects at rho
        z = self.o_z + self.slope[:, col] * rho[None, :]
        in_slab = (z >= 0.0) & (z <= cfg.obstacle_max_height_m)
        n_in = in_slab.sum(axis=0)
        n_nr = (in_slab & (self.state[:, col] == CellState.NO_RETURN)).sum(axis=0)
        dropout = (n_in == 0) | (n_nr > cfg.dropout_fraction * np.maximum(n_in, 1))
        first = self.rho_obs[col]
        out[dropout] = PointState.DROPOUT
        out[(rho > first + cfg.occlusion_tol_m) | (rho > self.rho_ground_limit[col])] = PointState.OCCLUDED
        out[np.abs(rho - first) <= cfg.occlusion_tol_m] = PointState.BLOCKED
        out[rho > self.max_range] = PointState.RANGE
        out[outside] = PointState.OUTSIDE
        out[rho < self.near_m] = PointState.NEAR
        return out


@dataclass(frozen=True)
class ViewMeasure:
    sight_distance_m: float             # from the front bumper
    limit: str                          # why the sight distance ends
    unknown_fraction: float
    n_cells: int                        # cells counted for the unknown fraction
    hazard_ahead: bool                  # an obstacle blocks the corridor within eval_range_m


def corridor_cells(corridor: Corridor, cfg: ViewConfig = ViewConfig()) -> tuple:
    """Stations and cells of the corridor: (s (S,), l (L,), points (S, L, 2)); s from the front bumper to eval_range_m (or the end)."""
    s_max = min(corridor.length_m, corridor.front_offset_m + cfg.eval_range_m)
    if abs(corridor.k) > 1e-9:
        s_max = min(s_max, 0.9 * math.pi / abs(corridor.k))                  # the arc turns back on itself after half a circle: nothing ahead there
    n_s = max(int(math.floor((s_max - corridor.front_offset_m) / cfg.station_step_m)), 1)
    s = corridor.front_offset_m + (np.arange(n_s) + 0.5) * cfg.station_step_m
    n_l = max(int(round(2.0 * corridor.half_width_m / cfg.lateral_step_m)) + 1, 2)
    lat = np.linspace(-corridor.half_width_m, corridor.half_width_m, n_l)
    k = corridor.k
    if abs(k) < 1e-9:
        centre, ang = np.stack([s, np.zeros_like(s)], axis=1), np.zeros_like(s)
    else:
        centre, ang = np.stack([np.sin(k * s) / k, (1.0 - np.cos(k * s)) / k], axis=1), k * s
    normal = np.stack([-np.sin(ang), np.cos(ang)], axis=1)
    return s, lat, centre[:, None, :] + lat[None, :, None] * normal[:, None, :]


def measure_view(view: ViewField, corridor: Corridor, cfg: ViewConfig | None = None) -> ViewMeasure:
    cfg = cfg or view.cfg
    s, lat, pts = corridor_cells(corridor, cfg)
    states = view.classify(pts.reshape(-1, 2)).reshape(pts.shape[:2])
    counted = states != PointState.NEAR
    unknown = np.isin(states, UNKNOWN_STATES)
    blocked = (states == PointState.BLOCKED).any(axis=1)
    n_cnt = counted.sum(axis=1)
    station_unknown = (unknown.sum(axis=1) >= cfg.station_unknown_fraction * np.maximum(n_cnt, 1)) & (n_cnt > 0)
    need = max(int(round(cfg.min_run_m / cfg.station_step_m)), 1)
    run_end = len(s)
    kind, edge = "clear", None
    for i in range(len(s)):
        if blocked[i]:
            kind, edge, run_end = "hazard", i, i
            break
        if station_unknown[i] and station_unknown[i:i + need].all() and i + need <= len(s):
            dominant = np.bincount(states[i][unknown[i]], minlength=7).argmax()
            kind, edge, run_end = PointState(int(dominant)).name.lower(), i, len(s)
            break
    if edge is None:
        sight = float(s[-1] + 0.5 * cfg.station_step_m - corridor.front_offset_m)
    else:
        sight = float(s[edge] - 0.5 * cfg.station_step_m - corridor.front_offset_m)
    upto = edge if kind == "hazard" else len(s)
    cells = counted[:upto]
    n_cells = int(cells.sum())
    frac = float(unknown[:upto][cells].mean()) if n_cells else 0.0
    return ViewMeasure(sight_distance_m=max(sight, 0.0), limit=kind, unknown_fraction=frac, n_cells=n_cells, hazard_ahead=kind == "hazard")