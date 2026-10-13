"""The visible surface of a track is given to the risk stage (step 7.3)."""
import numpy as np
import pytest

from server.tracking.outputs import track_output
from server.tracking.tracker import Tracker
from testing.tests.server.tracking.test_extent import seen
from testing.tests.server.tracking.test_tracker import DT


def test_the_output_has_the_corners_and_the_nearest_point_relative_to_the_track():
    tr = Tracker()
    for k in range(6):
        w = seen([20.0, -0.9], [24.5, 0.9], k * DT)
        tr.step([w], k * DT)
    out = track_output(tr.tracks[0])
    assert out.surface_offsets.shape == (5, 2)
    assert out.position + out.surface_offsets[4] == pytest.approx(w.nearest_xy)                       # the nearest point
    assert out.position + out.surface_offsets[:4] == pytest.approx(w.corners_xy)


def test_the_surface_moves_with_the_track_when_it_coasts():
    tr = Tracker()
    for k in range(10):
        x = 20.0 + 3.0 * k * DT
        tr.step([seen([x, -0.9], [x + 4.5, 0.9], k * DT)], k * DT)
    before = track_output(tr.tracks[0])
    nearest_before = before.position + before.surface_offsets[4]
    tr.tracks[0].filter.predict(before.t + 0.3)                                                       # coasting: the point moves with the velocity
    after = track_output(tr.tracks[0])
    moved = after.position + after.surface_offsets[4] - nearest_before
    assert moved == pytest.approx(after.position - before.position, abs=1e-9)                          # the surface moves rigidly with the track
    assert np.hypot(*moved) > 0.5                                                                       # and it did move (about 3 m/s * 0.3 s)
    assert after.surface_offsets == pytest.approx(before.surface_offsets)                              # offsets are relative to the point