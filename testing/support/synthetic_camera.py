"""A synthetic forward camera for tests: a calibration and simple images of boxes on a plain background.

Vehicle frame as everywhere: x forward, y left, z up. The camera sits at `mount` looking forward; its optical frame is
x right, y down, z forward (docs of calibration.py).
"""
from __future__ import annotations

import math

import numpy as np

from server.geometry.calibration import CameraCalibration, Mount


def make_camera(width: int = 640, height: int = 360, hfov_deg: float = 90.0, mount=(1.5, 0.0, 1.4)) -> CameraCalibration:
    fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    r = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    t = -r @ np.array(mount, dtype=float)
    vfov = 2 * math.degrees(math.atan((height / 2) / fx))
    return CameraCalibration(width, height, 30.0, "jpeg", fx, fx, width / 2, height / 2, hfov_deg, vfov,
                             Mount(*mount), r, t)


def box_rect(camera: CameraCalibration, x, y, length, width, height, z0=0.0):
    """Pixel rectangle (u0, v0, u1, v1) covering a box standing at (x, y), seen from the camera."""
    xs, ys, zs = [x - length / 2, x + length / 2], [y - width / 2, y + width / 2], [z0, z0 + height]
    corners = np.array([[a, b, c] for a in xs for b in ys for c in zs])
    u, v, _ = camera.project(corners)
    return float(u.min()), float(v.min()), float(u.max()), float(v.max())


def render(camera: CameraCalibration, boxes=(), vertical_bars=(), noise=1.5, seed=0, flat=False) -> np.ndarray:
    """RGB uint8 image. boxes: (x, y, length, width, height) drawn dark. vertical_bars: pixel columns (u0, u1) of full-height
    dark bars (clutter). flat = a plain grey image (no contrast at all)."""
    h, w = camera.height, camera.width
    img = np.full((h, w, 3), 128.0)
    if not flat:
        horizon = int(camera.cy)
        img[:horizon] = 185.0                                  # sky
        img[horizon:] = 105.0                                  # road
    for (x, y, length, width, height) in sorted(boxes, key=lambda b: -b[0]):
        u0, v0, u1, v1 = box_rect(camera, x, y, length, width, height)
        img[max(0, round(v0)):max(0, round(v1)), max(0, round(u0)):max(0, round(u1))] = 35.0
    for u0, u1 in vertical_bars:
        img[:, max(0, round(u0)):max(0, round(u1))] = 35.0
    img += np.random.default_rng(seed).normal(0.0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)