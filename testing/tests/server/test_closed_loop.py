"""End to end without Unity (step 7.15): the real perception, tracker and risk stage drive a simple car in ray-cast scenes."""
import pytest  # noqa: F401

from testing.support.closed_loop import Box, LoopConfig, run_loop


def first_emergency(r):
    """The rows around the first emergency in words (None without one), for the failure message."""
    i = next((k for k, q in enumerate(r.rows) if q["emergency"]), None)
    if i is None:
        return None
    return "\n" + "\n".join(f"t={q['t']:.1f} x={q['x']:.1f} v={q['v']:.2f} brake={q['brake_request']:.1f} emergency={q['emergency']} {q['detail']}"
                            for q in r.rows[max(0, i - 4):i + 2])


def test_an_empty_road_is_driven_without_braking():
    r = run_loop([], LoopConfig(start_speed_mps=10.0, duration_s=5.0))
    assert not r.collided and r.max_brake == 0.0 and not r.any_emergency
    assert r.final["v"] > 11.0 and r.final["risk_level"] == 0                          # accelerates to the desired speed


def test_a_stopped_car_in_the_lane_is_stopped_for_without_a_collision():
    r = run_loop([Box(80.0, 0.0)])
    assert not r.collided and r.stopped
    assert 0.5 < r.final["gap_m"] < 40.0                                              # in front of it, and not absurdly far
    assert not r.any_emergency, first_emergency(r)
    assert max((q["brake_request"] for q in r.rows if q["v"] >= 3.0), default=0.0) <= 6.0     # 74 m are enough for an ordinary stop; the last
    #                                                                                   # metres of a stop inside the margin are not judged


def test_a_narrow_pedestrian_sized_object_in_the_lane_is_stopped_for():
    r = run_loop([Box(35.0, 0.0, 0.5, 0.5, 1.7)], LoopConfig(start_speed_mps=8.0, desired_mps=8.0))
    assert not r.collided and r.stopped and r.final["gap_m"] > 0.3
    assert not r.any_emergency, first_emergency(r)                                      # the end of the stop uses the margin only


PARKED = ([Box(50.0, 3.5)], LoopConfig(start_speed_mps=10.0, desired_mps=10.0, duration_s=40.0))


def test_a_parked_car_beside_the_road_is_passed_without_a_collision_or_an_emergency():
    r = run_loop(*PARKED)
    assert not r.collided
    assert not r.any_emergency, first_emergency(r)
    assert r.final["x"] > 60.0 and not r.stopped                                      # it slowed to the floor speed, passed, and drove on


def test_the_time_per_scan_fits_the_budget_on_average():
    r = run_loop([Box(80.0, 0.0)], LoopConfig(duration_s=6.0))
    assert r.percentile_ms("stage_ms", 50.0) < 33.0                                    # the stage alone; perception is measured by the tool