"""Camera bearing measurements for tracks (step 6.6).

The LiDAR gives positions at 10 Hz; the camera gives a direction to an object at its own rate and time. For every reported
track (confirmed or coasting) and every camera frame:

  1. Predict the track to the capture time of the frame (the filter does that, also for a frame older than the track).
  2. Region of interest: the expected lateral extent of the object (the width of its last LiDAR cluster, standing on the
     ground, as tall as the track has been seen) is projected into the image; the ROI is that box plus a margin.
  3. Classical edge detection (the same Canny settings and the same one-clear-edge rule as the refinement of step 5.6)
     looks for exactly one clear vertical edge near each side of the expected extent. Both found: the object is in the
     image there. Anything else (no_edge, ambiguous, low_contrast, image_border, behind_camera, implausible width) gives no
     measurement: the camera never creates or deletes anything and never makes up a direction.
  4. The middle between the two edges is a pixel column; with the camera calibration that is a direction in the vehicle frame,
     and with the pose of the car at the capture time a bearing in the world frame, seen from the camera position.
  5. The bearing goes through a gate (squared normalised innovation, 1 degree of freedom) and an extended Kalman update
     (kalman.py). It corrects the position across the line of sight, the LiDAR keeps the range.

The standard deviation of a bearing is the pixel uncertainty of the middle of the edges (`sigma_px` / fx) plus a small
constant for the pose yaw and the calibration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from server.geometry.calibration import CameraCalibration
from server.geometry.frames import wrap_angle
from server.perception.camera_refinement import RefinementConfig, _find_edge


@dataclass(frozen=True)
class BearingConfig:
    enabled: bool = True
    sigma_px: float = 2.0               # pixel standard deviation of the middle of the two edges
    sigma_extra_rad: float = 0.002      # pose yaw and calibration error (added in quadrature)
    gate_chi2: float = 9.0              # 3 sigma, 1 degree of freedom
    margin_frac: float = 0.3            # ROI margin: this fraction of the expected width ...
    min_margin_px: float = 8.0          # ... but at least this many pixels
    min_width_px: float = 4.0           # the two edges must be at least this far apart
    width_ratio: tuple = (0.4, 3.0)     # the measured lateral width against the expected one: outside this it is implausible
    min_range_m: float = 2.0            # closer than this (from the camera) no measurement
    use_color: bool = True              # edges of every colour channel too (an object and its background can share a gray level)
    color_canny_factor: float = 0.6     # Canny thresholds of the colour channels: this fraction of the gray ones
    edges: RefinementConfig = field(default_factory=RefinementConfig)


@dataclass
class BearingMeasurement:
    track_id: int
    reason: str                         # "ok" or why there is no measurement
    theta: float | None = None          # bearing in the world frame, rad from the x axis, positive to the left
    sigma: float | None = None          # rad
    cam_xy: tuple | None = None         # camera position in the world frame
    u: tuple | None = None              # the two edges in pixels
    expected_u: tuple | None = None     # the pixel extent where the object was expected (for diagnostics)
    v: tuple | None = None              # its rows (top, bottom)
    roi: tuple | None = None            # the search region (c0, r0, c1, r1)


@dataclass
class CameraResult:
    track_id: int
    reason: str                         # "ok", out_of_order, same_time, or why nothing was applied
    innovation_rad: float | None = None
    nis: float | None = None


@dataclass
class FrameEdges:
    gray: np.ndarray
    edges: np.ndarray                   # bool: Canny of the blurred gray image and, with use_color, of each colour channel
    rgb: np.ndarray | None = None       # the frame, for the contrast test


# ---- geometry ----------------------------------------------------------------------------------------------------------
def camera_origin_vehicle(camera: CameraCalibration) -> np.ndarray:
    """Position of the camera in the vehicle frame (3,)."""
    return -camera.r_optical_from_vehicle.T @ camera.t_optical_from_vehicle


def pixel_bearing_vehicle(camera: CameraCalibration, u: float, v: float | None = None) -> float:
    """Direction of the pixel column u (at row v, default the principal row) in the vehicle frame: rad from the x axis,
    positive to the left (the right of the image is negative)."""
    v = camera.cy if v is None else v
    d_opt = np.array([(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, 1.0])
    d_veh = camera.r_optical_from_vehicle.T @ d_opt
    return math.atan2(d_veh[1], d_veh[0])


def camera_xy_world(camera: CameraCalibration, pose) -> tuple:
    o = camera_origin_vehicle(camera)
    c, s = math.cos(pose[3]), math.sin(pose[3])
    return (pose[0] + c * o[0] - s * o[1], pose[1] + s * o[0] + c * o[1])


def prepare_frame(rgb: np.ndarray, config: BearingConfig = BearingConfig()) -> FrameEdges:
    """Gray image and edge map of a camera frame, once for all tracks. A yellow box on a gray road is nearly invisible in the
    gray image (similar gray levels), so with `use_color` the edges of the red, green and blue channels are added."""
    rgb = np.ascontiguousarray(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    k = config.edges.blur_ksize | 1
    lo, hi = config.edges.canny_low, config.edges.canny_high
    edges = cv2.Canny(cv2.GaussianBlur(gray, (k, k), 0), lo, hi) > 0
    if config.use_color:
        blur = cv2.GaussianBlur(rgb, (k, k), 0)
        f = config.color_canny_factor
        for ch in range(3):
            edges |= cv2.Canny(np.ascontiguousarray(blur[..., ch]), lo * f, hi * f) > 0
    return FrameEdges(gray, edges, rgb)


def _contrast(frame: FrameEdges, r0, r1, c0, c1, config) -> float:
    if config.use_color and frame.rgb is not None:
        return max(float(frame.rgb[r0:r1, c0:c1, ch].std()) for ch in range(3))
    return float(frame.gray[r0:r1, c0:c1].std())


# ---- one track in one frame ---------------------------------------------------------------------------------------------
def measure_bearing(track, frame: FrameEdges, camera: CameraCalibration, pose, t: float,
                    config: BearingConfig = BearingConfig()) -> BearingMeasurement:
    """The bearing of `track` in the camera frame captured at time t with the car at `pose` (or the reason there is none)."""
    diag = {}

    def none(reason):
        return BearingMeasurement(track.id, reason, **diag)

    x, _ = track.filter.predicted(t)
    c, s = math.cos(pose[3]), math.sin(pose[3])
    d = x[:2] - np.array([pose[0], pose[1]])
    pv = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])                          # predicted position, vehicle frame
    origin = camera_origin_vehicle(camera)
    rel = pv - origin[:2]
    rng = float(np.hypot(*rel))
    if rel[0] <= 0.5 or rng < config.min_range_m:
        return none("behind_camera")
    lateral = np.array([-rel[1], rel[0]]) / rng
    w_exp = max(float(track.last_cluster.width_m), 0.5)
    h_exp = max(float(track.max_height_m), 0.5)
    pts = np.array([[*(pv + sgn * lateral * w_exp / 2.0), z] for sgn in (-1.0, 1.0) for z in (0.0, h_exp)])
    u, v, depth = camera.project(pts)
    if not (depth > 0.5).all():
        return none("behind_camera")
    lo_u, hi_u = float(u.min()), float(u.max())
    v_top, v_bot = float(v.min()), float(v.max())
    diag.update(expected_u=(lo_u, hi_u), v=(v_top, v_bot))
    ecfg = config.edges
    m = max(config.min_margin_px, config.margin_frac * (hi_u - lo_u))
    h_img, w_img = frame.gray.shape
    r0, r1 = int(max(0, math.floor(v_top - m))), int(min(h_img, math.ceil(v_bot + m)))
    c0, c1 = int(max(0, math.floor(lo_u - m))), int(min(w_img, math.ceil(hi_u + m)))
    diag.update(roi=(c0, r0, c1, r1))
    if r1 - r0 < 4 or c1 - c0 < 4 or lo_u - m < 2 or hi_u + m > w_img - 2:
        return none("image_border")
    if _contrast(frame, r0, r1, c0, c1, config) < ecfg.min_contrast:
        return none("low_contrast")
    edges = frame.edges[r0:r1, c0:c1]
    rows = slice(max(0, int(math.floor(v_top)) - r0), r1 - r0)
    profile = edges[rows].sum(axis=0) / max(1, edges[rows].shape[0])
    found = []
    for lo_side, edge_u in ((True, lo_u), (False, hi_u)):
        a, b = (edge_u - m, edge_u + 0.5 * m) if lo_side else (edge_u - 0.5 * m, edge_u + m)
        pos, reason = _find_edge(profile, int(math.floor(a)) - c0, int(math.ceil(b)) - c0, ecfg)
        if pos is None:
            return none(reason)
        found.append(pos + c0)
    new_lo, new_hi = found
    if new_hi - new_lo < config.min_width_px:
        return none("no_edge")
    width_m = (new_hi - new_lo) * float(np.median(depth)) / camera.fx
    ratio = width_m / w_exp
    if not config.width_ratio[0] <= ratio <= config.width_ratio[1]:
        return none("implausible")
    u_mid = 0.5 * (new_lo + new_hi)
    theta = wrap_angle(pose[3] + pixel_bearing_vehicle(camera, u_mid, 0.5 * (v_top + v_bot)))
    sigma = math.hypot(config.sigma_px / camera.fx, config.sigma_extra_rad)
    return BearingMeasurement(track.id, "ok", theta, sigma, camera_xy_world(camera, pose), (new_lo, new_hi), **diag)