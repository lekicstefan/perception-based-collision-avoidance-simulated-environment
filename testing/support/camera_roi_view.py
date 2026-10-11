"""Pictures of the camera bearing step (step 6.6): where did the tracker look in the camera frame, and what did it find?

Left: the camera frame. For every reported track: the search region (yellow), the extent where the object was expected (cyan
vertical lines), and when a bearing was measured the two edges that were found (green) and the direction of the track
(white tick). Label: track id and the result (ok, or the reason there is no bearing: no_edge, ambiguous, image_border,
behind_camera, low_contrast, implausible, gated). Right: the edge map (Canny) the search works on, with the same regions.
"""
from __future__ import annotations

import cv2
import numpy as np

from server.tracking.bearing import BearingConfig, prepare_frame

YELLOW, CYAN, GREEN, WHITE = (255, 220, 0), (0, 220, 255), (60, 230, 90), (255, 255, 255)
RED = (255, 90, 90)


def _label(img, text, org, colour):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)


def _draw(img, m, result_reason: str):
    if m.roi is not None:
        c0, r0, c1, r1 = m.roi
        cv2.rectangle(img, (c0, r0), (c1, r1), YELLOW, 1)
    if m.expected_u is not None and m.v is not None:
        for u in m.expected_u:
            cv2.line(img, (int(round(u)), int(m.v[0]) - 6), (int(round(u)), int(m.v[1]) + 6), CYAN, 1, cv2.LINE_AA)
    if m.u is not None and m.v is not None:
        for u in m.u:
            cv2.line(img, (int(round(u)), int(m.v[0]) - 10), (int(round(u)), int(m.v[1]) + 10), GREEN, 2, cv2.LINE_AA)
    if m.roi is not None:
        c0, r0, _, _ = m.roi
        _label(img, f"#{m.track_id} {result_reason}", (max(2, c0), max(12, r0 - 4)), GREEN if result_reason in ("ok", "same_time", "out_of_order") else RED)
    elif m.expected_u is None:
        pass


def draw_rois(rgb: np.ndarray, diagnostics: list, results: list, config: BearingConfig = BearingConfig(), t: float | None = None):
    """RGB image (frame with overlays | edge map with overlays). `diagnostics` and `results` come from Tracker.step_camera."""
    left = np.ascontiguousarray(rgb).copy()
    edges = prepare_frame(rgb, config).edges
    right = np.repeat((edges.astype(np.uint8) * 255)[..., None], 3, axis=2)
    reason = {r.track_id: r.reason for r in results}
    for m in diagnostics:
        shown = reason.get(m.track_id, m.reason)
        _draw(left, m, shown)
        _draw(right, m, shown)
    if t is not None:
        _label(left, f"t = {t:.2f} s   tracks looked at: {len(diagnostics)}", (6, 14), WHITE)
    return np.hstack([left, right])