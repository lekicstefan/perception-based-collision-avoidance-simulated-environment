"""The tracker overlay of the scan pictures (testing/support/track_view.py and scan_view.draw_tracks)."""
import numpy as np

from server.tracking.tracker import Tracker
from testing.support import scan_view, track_view
from testing.tests.server.tracking.test_extent import seen
from testing.tests.server.tracking.test_tracker import DT

POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _run(n=8):
    tr = Tracker()
    ws = []
    for k in range(n):
        w = seen([20.0, -0.9], [24.5, 0.9], k * DT)
        ws.append(w)
        res = tr.step([w], k * DT)
    return tr, res, ws[-1]


def test_the_overlay_has_a_track_a_link_and_a_headline():
    tr, res, w = _run()
    tracks, links, head = track_view.overlay(tr, res, [w], POSE)
    assert len(tracks) == 1 and len(links) == 1 and "confirmed" in head
    t = tracks[0]
    assert t["ellipse"].shape[1] == 2 and t["box"].shape == (4, 2) and t["label"].startswith("T")
    assert np.allclose(t["pos"], tr.tracks[0].position)                       # pose = identity: vehicle frame = world frame


def test_the_overlay_follows_the_pose():
    tr, res, w = _run()
    pose = (5.0, 0.0, 0.0, 0.0, 0.0, 0.0)                                      # the car is 5 m further: the track is 5 m nearer
    tracks, _, _ = track_view.overlay(tr, res, [w], pose)
    assert np.allclose(tracks[0]["pos"], tr.tracks[0].position - [5.0, 0.0])


def test_draw_tracks_paints_pixels():
    tr, res, w = _run()
    tracks, links, _ = track_view.overlay(tr, res, [w], POSE)
    canvas = np.zeros((600, 500, 3), dtype=np.uint8)
    scan_view.draw_tracks(canvas, lambda x, y: (int(250 - y * 10), int(580 - x * 10)), tracks, links, 10.0)
    assert canvas.any()