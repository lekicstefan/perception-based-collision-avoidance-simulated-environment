"""Camera edge refinement at LiDAR scan times (step 5.6). A toggle: RefinementConfig.enabled.

The LiDAR sees an object in cells 0.2 degrees wide and can miss its silhouette edges (grazing angles, dropouts). The
camera sees the edge. For every cluster:

  1. Take the camera frame closest in time to the scan (CameraBuffer, at most max_time_gap_s away) and move the cluster
     points from the vehicle frame of the scan to the vehicle frame of that frame with the two poses (the car moved).
  2. Project them into the image. Their horizontal extent, widened by half a LiDAR cell on each side, is the LiDAR extent
     in pixels. The region of interest (ROI) is that box plus a margin.
  3. Classical edge detection in the ROI: grayscale, blur, Canny. The number of edge pixels per column, over the rows of the
     object, shows vertical edges as peaks.
  4. For each side, look for exactly one clear edge near the LiDAR boundary (up to a margin outwards, half of it inwards).
     Found: the side takes the edge position. Anything else keeps the LiDAR boundary (fallback). The reasons:
        no_camera_frame, behind_camera, image_border, low_contrast (fog, night), no_edge, ambiguous (two comparable edges,
        for example clutter right next to the object).
  5. The refined extent gives a new lateral width and a lateral shift of the box centre (metres at the cluster's depth,
     depth / fx per pixel). apply_refinement() puts both into the ClusterGeometry.

Only the lateral extent is refined, never the range, and a refinement can only move a side by the margin, so a bad edge
cannot move an object far. The camera never creates a cluster.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace

import cv2
import numpy as np

from server.communication import protocol as P
from server.geometry.calibration import CameraCalibration
from server.perception.cluster_geometry import ClusterGeometry
from server.perception.range_image import LidarGeometry, RangeImage


@dataclass(frozen=True)
class RefinementConfig:
    enabled: bool = True
    max_time_gap_s: float = 0.02        # a camera frame further from the scan time than this is not used
    margin_frac: float = 0.3            # ROI margin: this fraction of the LiDAR width ...
    min_margin_px: float = 6.0          # ... but at least this many pixels
    blur_ksize: int = 5
    canny_low: float = 40.0
    canny_high: float = 120.0
    min_edge_frac: float = 0.4          # an edge column holds edge pixels on this fraction of the object's rows
    min_contrast: float = 6.0           # gray level standard deviation of the ROI below which it is "low contrast"
    ambiguity_ratio: float = 0.7        # a second edge at least this strong as the best one makes the side ambiguous


@dataclass(frozen=True)
class Refinement:
    label: int
    used_left: bool
    used_right: bool
    reason_left: str                    # "ok" or the reason for the fallback
    reason_right: str
    u_lidar: tuple                      # (left, right) pixel extent from the LiDAR, widened by half a cell
    u_refined: tuple                    # (left, right) after refinement (equal to u_lidar where a side fell back)
    width_lidar_m: float
    width_refined_m: float
    shift_left_m: float                 # lateral shift of the box centre, positive to the left

    @property
    def applied(self) -> bool:
        return self.used_left or self.used_right


def to_camera_vehicle(points: np.ndarray, pose_scan, pose_camera) -> np.ndarray:
    """Points in the vehicle frame at the scan time -> the vehicle frame at the camera capture time (x, y, yaw only)."""
    c, s = math.cos(pose_scan[3]), math.sin(pose_scan[3])
    wx = pose_scan[0] + c * points[..., 0] - s * points[..., 1]
    wy = pose_scan[1] + s * points[..., 0] + c * points[..., 1]
    dx, dy = wx - pose_camera[0], wy - pose_camera[1]
    c2, s2 = math.cos(pose_camera[3]), math.sin(pose_camera[3])
    return np.stack([c2 * dx + s2 * dy, -s2 * dx + c2 * dy, points[..., 2]], axis=-1)


class CameraBuffer:
    """The last few camera messages, decoded only when a frame is asked for."""

    def __init__(self, max_frames: int = 30):
        self._frames = deque(maxlen=max_frames)       # (t, header, raw message)
        self._decoded = {}

    def add(self, raw_message: bytes) -> None:
        h = P.parse_header(raw_message)
        self._frames.append((h.t, h, bytes(raw_message)))
        live = {f[1].seq for f in self._frames}
        for seq in [k for k in self._decoded if k not in live]:
            del self._decoded[seq]

    def nearest(self, t: float, max_gap_s: float):
        """(header, rgb image) of the frame closest to t, or None when none is within max_gap_s."""
        if not self._frames:
            return None
        ft, h, raw = min(self._frames, key=lambda f: abs(f[0] - t))
        if abs(ft - t) > max_gap_s:
            return None
        if h.seq not in self._decoded:
            _, payload = P.parse_camera(raw)
            self._decoded[h.seq] = P.camera_to_rgb(payload)
        return h, self._decoded[h.seq]


class CameraRefiner:
    def __init__(self, camera: CameraCalibration, geometry: LidarGeometry, config: RefinementConfig | None = None):
        self.camera = camera
        self.config = config or RefinementConfig()
        self.half_cell_px = 0.5 * float(geometry.alpha_h.mean()) * camera.fx

    # ------------------------------------------------------------------
    def refine_and_apply(self, geoms: list, img: RangeImage, cam_header, rgb: np.ndarray | None):
        """(geometries with the refinement applied, list of Refinement). Disabled or without a frame: unchanged."""
        if not self.config.enabled:
            return geoms, []
        refs = self.refine(geoms, img, cam_header, rgb)
        return [apply_refinement(g, r) for g, r in zip(geoms, refs)], refs

    def refine(self, geoms: list, img: RangeImage, cam_header, rgb: np.ndarray | None) -> list:
        gray = blur = None
        if rgb is not None:
            gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
            k = self.config.blur_ksize | 1
            blur = cv2.GaussianBlur(gray, (k, k), 0)
        return [self._one(g, img, cam_header, gray, blur) for g in geoms]

    # ------------------------------------------------------------------
    def _fallback(self, g, reason, u=(0.0, 0.0), width=0.0) -> Refinement:
        return Refinement(g.label, False, False, reason, reason, u, u, width, width, 0.0)

    def _one(self, g: ClusterGeometry, img: RangeImage, cam_header, gray, blur) -> Refinement:
        cfg, cam = self.config, self.camera
        if gray is None or cam_header is None:
            return self._fallback(g, "no_camera_frame")
        pts = to_camera_vehicle(g.points, img.pose, cam_header.pose)
        u, v, depth = cam.project(pts)
        if not (depth > 0.5).all():
            return self._fallback(g, "behind_camera")
        z = float(np.median(depth))
        lo_u, hi_u = float(u.min()) - self.half_cell_px, float(u.max()) + self.half_cell_px
        v_top, v_bot = float(v.min()), float(v.max())
        m = max(cfg.min_margin_px, cfg.margin_frac * (hi_u - lo_u))
        w_lidar = (hi_u - lo_u) * z / cam.fx
        base = dict(u=(lo_u, hi_u), width=w_lidar)
        h_img, w_img = gray.shape
        r0, r1 = int(max(0, math.floor(v_top - m))), int(min(h_img, math.ceil(v_bot + m)))
        c0, c1 = int(max(0, math.floor(lo_u - m))), int(min(w_img, math.ceil(hi_u + m)))
        if r1 - r0 < 4 or c1 - c0 < 4:
            return self._fallback(g, "image_border", **base)
        roi = blur[r0:r1, c0:c1]
        if float(gray[r0:r1, c0:c1].std()) < cfg.min_contrast:
            return self._fallback(g, "low_contrast", **base)
        edges = cv2.Canny(roi, cfg.canny_low, cfg.canny_high) > 0
        rows = slice(max(0, int(math.floor(v_top)) - r0), r1 - r0)                   # from the top of the object to the ROI bottom
        profile = edges[rows].sum(axis=0) / max(1, edges[rows].shape[0])               # edge pixels per column, 0 ... 1

        sides, reasons, new = [], [], []
        for lo_side, edge_u in ((True, lo_u), (False, hi_u)):
            out, inn = m, 0.5 * m
            a, b = (edge_u - out, edge_u + inn) if lo_side else (edge_u - inn, edge_u + out)
            if a < 2 or b > w_img - 2:                                              # the window leaves the image
                sides.append(False); reasons.append("image_border"); new.append(edge_u); continue
            pos, reason = _find_edge(profile, int(math.floor(a)) - c0, int(math.ceil(b)) - c0, cfg)
            if pos is None:
                sides.append(False); reasons.append(reason); new.append(edge_u)
            else:
                sides.append(True); reasons.append("ok"); new.append(pos + c0)
        new_lo, new_hi = new
        if new_hi - new_lo < 2.0:                                                   # the sides crossed: do not trust either
            return self._fallback(g, "no_edge", **base)
        width = (new_hi - new_lo) * z / cam.fx
        shift = -((new_lo + new_hi) - (lo_u + hi_u)) / 2.0 * z / cam.fx             # image right = vehicle right
        return Refinement(g.label, sides[0], sides[1], reasons[0], reasons[1], (lo_u, hi_u), (new_lo, new_hi),
                          w_lidar, width, shift)


def _find_edge(profile: np.ndarray, lo: int, hi: int, cfg: RefinementConfig):
    """(column position of the one clear edge in profile[lo:hi + 1], "ok") or (None, reason)."""
    lo, hi = max(lo, 0), min(hi, len(profile) - 1)
    if hi <= lo:
        return None, "image_border"
    vals = profile[lo:hi + 1]
    strong = np.flatnonzero(vals >= cfg.min_edge_frac)
    if len(strong) == 0:
        return None, "no_edge"
    runs, start = [], 0
    for i in range(1, len(strong) + 1):                                              # runs of columns, gaps of 1 allowed
        if i == len(strong) or strong[i] - strong[i - 1] > 2:
            runs.append(strong[start:i])
            start = i
    strengths = [float(vals[r].max()) for r in runs]
    best = int(np.argmax(strengths))
    if any(s >= cfg.ambiguity_ratio * strengths[best] for k, s in enumerate(strengths) if k != best):
        return None, "ambiguous"
    r = runs[best]
    w = vals[r]
    return float((r * w).sum() / w.sum()) + lo + 0.5, "ok"


def apply_refinement(g: ClusterGeometry, r: Refinement) -> ClusterGeometry:
    """The geometry with the refined lateral width and the box centre moved sideways. Unchanged if nothing was refined."""
    if not r.applied:
        return g
    v = np.array([-math.sin(g.bearing), math.cos(g.bearing)])                        # left of the line of sight
    return replace(g, width_m=r.width_refined_m, centre_xy=g.centre_xy + v * r.shift_left_m)