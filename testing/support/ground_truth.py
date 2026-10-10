"""Ground truth of a development recording, seen from the ego vehicle (evaluation only, never used by the processor).

Files read from the recording folder (written by GroundTruthLogger):
  ego.csv          t, x, y, z, yaw_rad, pitch_rad, speed, ...         true ego pose in the start frame, every physics step
  hazards.csv      t, id, x, y, z, yaw_rad, vx, vy, speed, length, width, height     box centre in the start frame
  hazard_info.csv  id, name, label, length, width, height, colliders
  visibility.csv   t, scan, id, hits_geometric, hits_returned, min_range_m, centre_range_m, azimuth_deg, in_view
                   one row per hazard and LiDAR scan: how many cells hit it, and how many of those returned

The evaluation is done in the vehicle frame of the scan (x forward, y left): truth positions are moved there with the TRUE
ego pose, so pose noise does not mix into the perception error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True, eq=False)
class TruthObject:
    id: int
    label: str
    centre: np.ndarray               # (2,) vehicle frame
    yaw: float                       # heading relative to the ego vehicle (rad)
    length: float
    width: float
    height: float
    hits_returned: int | None        # LiDAR cells that hit it and returned; None when the recording has no visibility row
    in_view: bool

    def _to_local(self, xy: np.ndarray) -> np.ndarray:
        """Vehicle frame points (n, 2) -> frame of the box (x along its length)."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        d = np.asarray(xy, dtype=float) - self.centre
        return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], axis=-1)

    def footprint(self) -> np.ndarray:
        """(4, 2) corners in the vehicle frame."""
        hl, hw = self.length / 2, self.width / 2
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
        return self.centre + local @ np.array([[c, s], [-s, c]])

    def contains(self, points_xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
        p = self._to_local(points_xy)
        return (np.abs(p[..., 0]) <= self.length / 2 + margin) & (np.abs(p[..., 1]) <= self.width / 2 + margin)

    def distance_from(self, origin_xy) -> float:
        """Ground-plane distance from a point to the box (0 inside)."""
        a, b = self._to_local(np.asarray(origin_xy, dtype=float)[None])[0]
        return float(math.hypot(max(abs(a) - self.length / 2, 0.0), max(abs(b) - self.width / 2, 0.0)))

    def lateral_extent(self, origin_xy) -> float:
        """Width of the box as seen from the point: its extent across the line of sight."""
        to = self.centre - np.asarray(origin_xy, dtype=float)
        u = to / np.linalg.norm(to)
        v = np.array([-u[1], u[0]])
        proj = (self.footprint() - np.asarray(origin_xy, dtype=float)) @ v
        return float(proj.max() - proj.min())


class GroundTruth:
    def __init__(self, ego: pd.DataFrame, hazards: pd.DataFrame, info: pd.DataFrame, visibility: pd.DataFrame):
        self.ego, self.hazards, self.info, self.visibility = ego, hazards, info, visibility
        self._ego_t = ego["t"].to_numpy()
        self._ego_yaw = np.unwrap(ego["yaw_rad"].to_numpy())
        self._haz = {int(i): g.sort_values("t") for i, g in hazards.groupby("id")}
        self._vis = {int(i): g.sort_values("t") for i, g in visibility.groupby("id")}
        self._label = {int(r["id"]): str(r["label"]) for _, r in info.iterrows()}

    @classmethod
    def load(cls, folder) -> "GroundTruth":
        f = Path(folder)
        return cls(pd.read_csv(f / "ego.csv"), pd.read_csv(f / "hazards.csv"), pd.read_csv(f / "hazard_info.csv"),
                   pd.read_csv(f / "visibility.csv"))

    @classmethod
    def exists(cls, folder) -> bool:
        return all((Path(folder) / n).exists() for n in ("ego.csv", "hazards.csv", "hazard_info.csv", "visibility.csv"))

    def ego_at(self, t: float):
        """(x, y, yaw) of the ego vehicle at time t in the start frame."""
        return (float(np.interp(t, self._ego_t, self.ego["x"])), float(np.interp(t, self._ego_t, self.ego["y"])),
                float(np.interp(t, self._ego_t, self._ego_yaw)))

    def objects_at(self, t: float, vis_tolerance_s: float = 0.006) -> list:
        """Every hazard at time t as a TruthObject in the vehicle frame of the ego at t."""
        ex, ey, eyaw = self.ego_at(t)
        c, s = math.cos(eyaw), math.sin(eyaw)
        out = []
        for hid, g in self._haz.items():
            gt = g["t"].to_numpy()
            if t < gt[0] - 1e-6 or t > gt[-1] + 1e-6:
                continue
            hx, hy = float(np.interp(t, gt, g["x"])), float(np.interp(t, gt, g["y"]))
            hyaw = float(np.interp(t, gt, np.unwrap(g["yaw_rad"].to_numpy())))
            dx, dy = hx - ex, hy - ey
            centre = np.array([c * dx + s * dy, -s * dx + c * dy])
            hits, in_view = None, False
            v = self._vis.get(hid)
            if v is not None:
                i = int(np.argmin(np.abs(v["t"].to_numpy() - t)))
                if abs(float(v["t"].iloc[i]) - t) <= vis_tolerance_s:
                    hits, in_view = int(v["hits_returned"].iloc[i]), bool(v["in_view"].iloc[i])
            out.append(TruthObject(hid, self._label.get(hid, str(hid)), centre, hyaw - eyaw, float(g["length"].iloc[0]),
                                   float(g["width"].iloc[0]), float(g["height"].iloc[0]), hits, in_view))
        return out