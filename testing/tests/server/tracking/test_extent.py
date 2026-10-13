"""Size memory of the tracks (step 6.11)."""
import math

import numpy as np
import pytest

from server.tracking.extent import ExtentConfig, centre_with_extent, depth_along, grow, hull, initial_extent
from server.tracking.measurement import WorldCluster
from server.tracking.tracker import Tracker, TrackerConfig
from testing.tests.server.tracking.test_tracker import DT

PRIOR = 1.0


def seen(lo, hi, t, sensor=(0.0, 0.0)):
    """A cluster that shows the part [lo, hi] (world, axis-aligned) of an object to a sensor, built like the perception does it:
    the centre is the nearest depth + max(visible depth, prior) / 2 along the line of sight, and the middle across it."""
    lo, hi, s = np.array(lo, float), np.array(hi, float), np.array(sensor, float)
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    mid = 0.5 * (lo + hi)
    u = (mid - s) / np.hypot(*(mid - s))
    v = np.array([-u[1], u[0]])
    pu, pv = (corners - s) @ u, (corners - s) @ v
    depth = float(pu.max() - pu.min())
    centre = s + u * (pu.min() + max(depth, PRIOR) / 2.0) + v * (pv.min() + pv.max()) / 2.0
    nearest = corners[int(np.argmin(pu))]                                                  # a real point of the object (inside the hull)
    return WorldCluster(t=t, pose=(0.0,) * 6, label=0, n_cells=50, nearest_xy=nearest, centre_xy=centre, centroid_xy=mid, sensor_xy=s,
                        bearing=math.atan2(u[1], u[0]), range_m=float(pu.min()), corners_xy=corners, width_m=float(pv.max() - pv.min()),
                        depth_m=depth, height_m=1.5, top_m=1.5)


def test_the_hull_is_the_box_of_the_visible_part():
    lo, hi = hull(seen([20.0, -0.9], [20.3, 0.9], 0.0))
    assert lo == pytest.approx([20.0, -0.9]) and hi == pytest.approx([20.3, 0.9])


def test_the_initial_extent_is_the_hull_with_the_absolute_cap():
    assert initial_extent(seen([20.0, -0.9], [20.3, 0.9], 0.0)) == pytest.approx([0.3, 1.8])
    assert initial_extent(seen([20.0, -0.9], [30.0, 0.9], 0.0), ExtentConfig(max_extent_m=8.0)) == pytest.approx([8.0, 1.8])


def test_the_depth_is_the_extent_of_the_remembered_box_along_the_line_of_sight():
    assert depth_along([4.5, 1.8], 0.0) == pytest.approx(4.5) and depth_along([4.5, 1.8], math.pi / 2) == pytest.approx(1.8)
    assert depth_along([0.3, 0.4], 0.3) == pytest.approx(1.0)                                  # never below the prior
    assert depth_along([4.5, 1.8], math.radians(45)) == pytest.approx(math.sqrt(0.5) * 6.3)


def test_without_memory_the_centre_is_the_one_of_perception():
    for w in (seen([20.0, -0.9], [20.3, 0.9], 0.0), seen([20.0, -0.9], [21.5, 0.9], 0.0), seen([-3.0, 8.0], [-2.5, 9.0], 0.0)):
        assert centre_with_extent(w, initial_extent(w)) == pytest.approx(w.centre_xy)


def test_a_remembered_size_moves_the_centre_along_the_line_of_sight_only():
    front = seen([20.0, -0.9], [20.3, 0.9], 0.0)                                              # the front of a car 4.5 m long
    assert centre_with_extent(front, [4.5, 1.8]) == pytest.approx([22.25, 0.0])
    side = seen([-2.25, 10.0], [2.25, 10.3], 0.0)                                             # its side, from the left
    assert centre_with_extent(side, [4.5, 1.8]) == pytest.approx([0.0, 10.9])
    assert centre_with_extent(front, [4.5, 1.8])[1] == pytest.approx(front.centre_xy[1])       # nothing across the line of sight


def test_the_extent_grows_with_a_cap_and_never_shrinks():
    cfg = ExtentConfig(max_growth_m=1.5, max_extent_m=8.0)
    big = seen([20.0, -0.9], [24.5, 0.9], 0.0)
    e = np.array([0.3, 1.8])
    for want in (1.8, 3.3, 4.5, 4.5):
        e = grow(e, big, cfg)
        assert e[0] == pytest.approx(want) and e[1] == pytest.approx(1.8)
    assert grow(e, seen([20.0, -0.9], [20.3, 0.9], 0.0), cfg) == pytest.approx(e)                 # a sliver does not shrink it
    assert grow([7.5, 1.8], seen([20.0, -0.9], [40.0, 0.9], 0.0), cfg)[0] == pytest.approx(8.0)       # the absolute cap


def _run(cfg, phases):
    tr = Tracker(cfg)
    k = 0
    for lo_hi, n in phases:
        for _ in range(n):
            tr.step([seen(*lo_hi, k * DT)], k * DT)
            k += 1
    return tr


PARKED = [(([20.0, -0.9], [20.3, 0.9]), 6), (([20.0, -0.9], [24.5, 0.9]), 10), (([20.0, -0.9], [20.3, 0.9]), 14)]       # centre (22.25, 0)


def test_a_partly_seen_object_keeps_its_centre_and_gets_no_velocity_with_the_memory():
    tr = _run(TrackerConfig(), PARKED)
    t = next(x for x in tr.tracks if x.hits > 20)
    assert np.hypot(*(t.position - [22.25, 0.0])) < 0.4 and t.filter.speed < 0.3 and np.round(t.extent, 2) == pytest.approx([4.5, 1.8])


def test_without_the_memory_the_centre_sits_near_the_visible_face():
    tr = _run(TrackerConfig(extent=ExtentConfig(enabled=False)), PARKED)
    t = max(tr.tracks, key=lambda x: x.last_hit_t)
    assert np.hypot(*(t.position - [22.25, 0.0])) > 1.0


def test_the_growth_of_the_memory_is_not_read_as_velocity():
    tr = Tracker()
    speeds = []
    for k in range(40):
        lo_hi = (([20.0, -0.9], [20.3, 0.9]) if k < 6 else ([20.0, -0.9], [24.5, 0.9]))
        tr.step([seen(*lo_hi, k * DT)], k * DT)
        speeds.append(tr.tracks[0].filter.speed)
    assert len(tr.tracks) == 1 and max(speeds[6:]) < 0.8 and speeds[-1] < 0.2           # one track throughout: the change was absorbed


def test_a_moving_object_still_moves():
    tr = Tracker()
    for k in range(40):
        x = 20.0 + 3.0 * k * DT                                                      # 3 m/s away from the sensor, full view
        tr.step([seen([x, -0.9], [x + 4.5, 0.9], k * DT)], k * DT)
    assert len(tr.tracks) == 1 and tr.tracks[0].filter.velocity[0] == pytest.approx(3.0, abs=0.4)


def test_a_small_object_is_not_changed_by_the_memory():
    on, off = Tracker(), Tracker(TrackerConfig(extent=ExtentConfig(enabled=False)))
    for k in range(20):
        for tr in (on, off):
            tr.step([seen([15.0, 2.0], [15.4, 2.5], k * DT)], k * DT)
    assert on.tracks[0].position == pytest.approx(off.tracks[0].position, abs=1e-6)


def test_other_points_are_not_touched_by_the_memory():
    tr = Tracker(TrackerConfig(point="nearest"))
    tr.step([seen([20.0, -0.9], [20.3, 0.9], 0.0)], 0.0)
    assert tr.tracks[0].extent is None and tr.tracks[0].position == pytest.approx([20.0, -0.9])


def test_one_scan_with_the_cluster_of_another_object_does_not_inflate_the_memory():
    tr = Tracker()
    for k in range(30):
        lo_hi = ([20.0, -0.25], [21.2, 0.25]) if k == 15 else ([20.0, -0.25], [20.1, 0.25])             # a person; once a 2 m cluster
        tr.step([seen(*lo_hi, k * DT)], k * DT)
    assert max(tr.tracks, key=lambda x: x.hits).extent[0] < 0.2


def test_a_size_seen_in_three_scans_is_remembered():
    tr = Tracker()
    for k in range(30):
        lo_hi = ([20.0, -0.9], [24.5, 0.9]) if k >= 10 else ([20.0, -0.9], [20.3, 0.9])
        tr.step([seen(*lo_hi, k * DT)], k * DT)
    assert tr.tracks[0].extent[0] == pytest.approx(4.5)


def test_persistence_of_one_scan_is_the_old_behaviour():
    cfg = ExtentConfig(persist_scans=1)
    e = grow([0.3, 1.8], seen([20.0, -0.9], [22.0, 0.9], 0.0), cfg, [np.array([2.0, 1.8])])
    assert e[0] == pytest.approx(1.8)                                                              # one scan, the cap of 1.5 m


def test_an_extent_that_includes_the_visible_hull_gives_the_centre_of_perception():
    big = seen([20.0, -0.9], [24.5, 0.9], 0.0)
    lo, hi = hull(big)
    assert centre_with_extent(big, np.maximum([0.3, 1.8], hi - lo)) == pytest.approx(big.centre_xy)     # what the tracker uses: memory or hull