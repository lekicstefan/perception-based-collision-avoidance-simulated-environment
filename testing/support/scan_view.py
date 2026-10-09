"""Pictures of what the LiDAR sees, for looking at a scan (used by testing/tools/perception/render_scan.py).

  range panel   the range image, one cell per block. Every cluster has its own colour; inside a cluster closer cells are
                lighter and paler, farther cells darker (shading is relative to that cluster). Ground is tan, cells of
                discarded (too small) clusters are dark red, no return is black, max range or sky is dark blue.
  top view      the same cells seen from above (x forward is up, y left is left), ego car, range rings, the nearest point
                (white cross) and the corrected box centre (yellow ring) of every cluster.
  camera panel  the camera frame closest in time with the projected LiDAR extent (blue) and the refined extent (green).
"""
from __future__ import annotations

import colorsys
import math

import cv2
import numpy as np

from server.perception.pipeline import PerceptionResult
from server.perception.range_image import CellState

GROUND = np.array([128, 110, 82])
DISCARDED = np.array([110, 30, 30])
NO_RETURN = np.array([0, 0, 0])
MAX_RANGE = np.array([14, 20, 48])
BACKGROUND = (24, 24, 28)


def cluster_hue(k: int) -> float:
    return (k * 0.61803398875 + 0.07) % 1.0                 # golden ratio steps: neighbours in k are far apart in hue


def shade(hue: float, closeness: np.ndarray) -> np.ndarray:
    """RGB (n, 3) uint8 for closeness in [0, 1] (1 = closest of the cluster): lighter and paler when close, darker when far."""
    c = np.clip(np.asarray(closeness, dtype=float), 0.0, 1.0)
    rgb = [colorsys.hsv_to_rgb(hue, 1.0 - 0.55 * ci, 0.30 + 0.70 * ci) for ci in c]
    return (np.array(rgb) * 255).round().astype(np.uint8)


def cluster_closeness(result: PerceptionResult, min_span_m: float = 0.5) -> np.ndarray:
    """Closeness (rows, cols) of every cell that belongs to a cluster, NaN elsewhere. Relative to the cluster; a cluster
    whose depth span is below min_span_m is shaded as if it were that deep, so a flat face is not stretched into noise."""
    out = np.full(result.image.shape, np.nan)
    rng = result.image.range_m
    for c in result.segmentation.clusters:
        r = rng[c.rows, c.cols]
        span = max(float(r.max() - r.min()), min_span_m)
        out[c.rows, c.cols] = 1.0 - (r - r.min()) / span
    return out


def cell_colours(result: PerceptionResult) -> np.ndarray:
    """(rows, cols, 3) uint8 colour of every cell."""
    img = result.image
    col = np.zeros(img.shape + (3,), dtype=np.uint8)
    col[img.state == CellState.NO_RETURN] = NO_RETURN
    col[img.state == CellState.MAX_RANGE] = MAX_RANGE
    cand = result.candidates
    col[cand] = DISCARDED                                                      # replaced below for kept clusters
    gr = result.ground
    rng = np.where(gr, img.range_m, np.nan)
    if gr.any():
        f = 1.0 - np.clip(rng[gr] / max(float(np.nanmax(rng)), 1.0), 0, 1) * 0.6        # ground fades with distance
        col[gr] = (GROUND[None, :] * f[:, None]).astype(np.uint8)
    closeness = cluster_closeness(result)
    for c in result.segmentation.clusters:
        col[c.rows, c.cols] = shade(cluster_hue(c.label), closeness[c.rows, c.cols])
    return col


def _text(img, s, org, scale=0.45, colour=(235, 235, 235)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def range_panel(result: PerceptionResult, calibration, sx: int = 3, sy: int = 14) -> np.ndarray:
    col = cell_colours(result)
    rows, cols = col.shape[:2]
    big = cv2.resize(col, (cols * sx, rows * sy), interpolation=cv2.INTER_NEAREST)
    pad_l, pad_t, pad_b = 46, 22, 24
    canvas = np.full((rows * sy + pad_t + pad_b, cols * sx + pad_l + 8, 3), BACKGROUND, dtype=np.uint8)
    canvas[pad_t:pad_t + rows * sy, pad_l:pad_l + cols * sx] = big
    lid = calibration.lidar
    for i in range(0, rows, max(1, rows // 8)):                                   # beam elevations on the left
        _text(canvas, f"{lid.elevations_deg[i]:+.0f}", (4, pad_t + i * sy + sy - 3), 0.38)
    az = lid.azimuths_deg
    for target in range(-int(abs(az).max()) // 20 * 20, int(abs(az).max()) + 1, 20):      # azimuth ticks below
        j = int(np.argmin(np.abs(az - target)))
        x = pad_l + j * sx
        cv2.line(canvas, (x, pad_t + rows * sy), (x, pad_t + rows * sy + 4), (200, 200, 200), 1)
        _text(canvas, f"{target:+d}", (x - 12, pad_t + rows * sy + 18), 0.38)
    for c in result.segmentation.clusters:                                         # labels
        g = next((q for q in result.clusters if q.label == c.label), None)
        top = int(c.rows.min())
        x = pad_l + int((c.cols.min() + c.cols.max()) / 2 * sx)
        tag = f"#{c.label} {c.range_m:.0f}m" + (f" {g.width_m:.1f}x{g.height_m:.1f}" if g is not None else "")
        _text(canvas, tag, (max(pad_l, x - 30), max(14, pad_t + top * sy - 3)), 0.4, tuple(int(v) for v in shade(cluster_hue(c.label), [0.9])[0]))
    counts = result.image.counts()
    _text(canvas, f"t={result.image.t:.2f}s  clusters={len(result.segmentation.clusters)} (discarded {result.segmentation.n_discarded})"
                  f"  ground={int(result.ground.sum())}  no return={counts['no_return']}  max range={counts['max_range']}",
          (pad_l, 14), 0.42)
    return canvas


def top_view(result: PerceptionResult, calibration, geometry, view_range_m: float = 50.0, half_width_m: float = 25.0,
             px_per_m: float = 10.0) -> np.ndarray:
    h, w = int(view_range_m * px_per_m) + 30, int(2 * half_width_m * px_per_m)
    canvas = np.full((h, w, 3), BACKGROUND, dtype=np.uint8)

    def pix(x, y):
        return int(round(w / 2 - y * px_per_m)), int(round(h - 20 - x * px_per_m))

    for r in range(10, int(view_range_m) + 1, 10):                                 # range rings
        cv2.circle(canvas, pix(0, 0), int(r * px_per_m), (60, 60, 66), 1, cv2.LINE_AA)
        _text(canvas, f"{r} m", (pix(0, 0)[0] + 4, pix(r, 0)[1] - 3), 0.38, (150, 150, 160))
    pts = geometry.points_vehicle(result.image)
    col = cell_colours(result)
    order = np.argsort(-np.nan_to_num(result.image.range_m, nan=-1).ravel())          # far first, near cells on top
    flat_p, flat_c = pts.reshape(-1, 3), col.reshape(-1, 3)
    valid = result.image.valid.ravel()
    for i in order:
        if not valid[i]:
            continue
        x, y = pix(flat_p[i, 0], flat_p[i, 1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(canvas, (x, y), 2, tuple(int(v) for v in flat_c[i]), -1)
    e = calibration.ego                                                            # the ego car
    p0, p1 = pix(e.length - e.rear_overhang, e.width / 2), pix(-e.rear_overhang, -e.width / 2)
    cv2.rectangle(canvas, p0, p1, (235, 235, 235), 1, cv2.LINE_AA)
    for g in result.clusters:
        nx, ny = pix(g.nearest_point[0], g.nearest_point[1])
        cv2.drawMarker(canvas, (nx, ny), (255, 255, 255), cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
        cx, cy = pix(g.centre_xy[0], g.centre_xy[1])
        cv2.circle(canvas, (cx, cy), 7, (60, 220, 255), 1, cv2.LINE_AA)
        _text(canvas, f"#{g.label}", (cx + 9, cy - 6), 0.4)
    _text(canvas, "top view: nearest point +, corrected centre o", (6, 14), 0.42)
    return canvas


def camera_panel(result: PerceptionResult, rgb: np.ndarray) -> np.ndarray:
    out = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    h = out.shape[0]
    for r in result.refinements:
        for (u0, u1), colour in ((r.u_lidar, (60, 140, 255)), (r.u_refined, (60, 230, 90))):
            if u1 > u0:
                for u in (u0, u1):
                    cv2.line(out, (int(round(u)), 0), (int(round(u)), h - 1), colour, 1)
        _text(out, f"#{r.label} {r.reason_left}/{r.reason_right}", (int(r.u_lidar[0]), 14 + 14 * (r.label % 8)), 0.4)
    _text(out, "camera: LiDAR extent blue, refined extent green", (6, h - 8), 0.42)
    return out


def compose(panels: list) -> np.ndarray:
    """The first panel (the range image) on top, the others side by side below it."""
    top, rest = panels[0], panels[1:]
    if not rest:
        return top
    height = max(p.shape[0] for p in rest)
    row = np.hstack([np.pad(p, ((0, height - p.shape[0]), (0, 0), (0, 0)), constant_values=24) for p in rest])
    width = max(top.shape[1], row.shape[1])
    pad = lambda im: np.pad(im, ((0, 0), (0, width - im.shape[1]), (0, 0)), constant_values=24)
    return np.vstack([pad(top), pad(row)])


def save_rgb(path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))