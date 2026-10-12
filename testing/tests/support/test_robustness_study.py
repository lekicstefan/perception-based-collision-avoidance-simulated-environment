"""Robustness study on the synthetic recording (step 6.10)."""
import numpy as np
import pytest

from testing.support.association_study import scan_measurements
from testing.support.robustness_study import (COLUMNS, evaluate, perturb_scans, rate_hz, run_scenarios, scenarios, shift_cluster,
                                              thin_scans)
from testing.support.synthetic_recording import write_recording
from testing.tools.tracking.robustness_check import main


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_recording(tmp_path_factory.mktemp("rec") / "synthetic")


@pytest.fixture(scope="module")
def scans(folder):
    return scan_measurements(folder, use_camera=True)


def _first_cluster(scans):
    return next(w for _, items in scans for w, _ in items)


def test_a_yaw_error_turns_points_about_the_car_and_a_position_error_shifts_them(scans):
    w = _first_cluster(scans)
    c = np.array(w.pose[:2])
    r = np.hypot(*(w.centre_xy - c))
    a = shift_cluster(w, [0.0, 0.0], 0.01)
    assert np.hypot(*(a.centre_xy - w.centre_xy)) == pytest.approx(0.01 * r, rel=0.01)
    assert np.hypot(*(a.centre_xy - c)) == pytest.approx(r, rel=1e-3) and a.bearing == pytest.approx(w.bearing + 0.01)
    b = shift_cluster(w, [0.3, -0.1], 0.0)
    assert b.centre_xy - w.centre_xy == pytest.approx([0.3, -0.1]) and b.corners_xy.shape == (4, 2) and b.pose[0] == pytest.approx(w.pose[0] + 0.3)


def test_no_perturbation_changes_nothing(scans):
    p = perturb_scans(scans)
    assert all(np.allclose(w.centre_xy, v.centre_xy) for (_, a), (_, b) in zip(scans, p) for (w, _), (v, _) in zip(a, b))


def test_the_bias_is_the_same_in_every_scan_and_the_noise_is_not(scans):
    b = perturb_scans(scans, bias_pos=0.5)
    shifts = {round(float((v.centre_xy - w.centre_xy)[0]), 6) for (_, a), (_, c) in zip(scans, b) for (w, _), (v, _) in zip(a, c)}
    assert shifts == {0.5}
    n = perturb_scans(scans, pos_sigma=0.2, seed=1)
    s = [float((c[0][0].centre_xy - a[0][0].centre_xy)[0]) for (_, a), (_, c) in zip(scans, n) if a]
    assert np.std(s) > 0.05


def test_thinning_keeps_every_nth_scan_and_the_drop_loses_some(scans):
    assert len(thin_scans(scans, every=2)) == (len(scans) + 1) // 2 and thin_scans(scans, every=2)[0] is scans[0]
    d = thin_scans(scans, drop=0.5, seed=3)
    assert 0.2 * len(scans) < len(d) < 0.8 * len(scans) and d[0] is scans[0]
    assert rate_hz(thin_scans(scans, every=2)) == pytest.approx(rate_hz(scans) / 2, rel=0.05)


def test_the_scenarios_start_with_the_baseline_and_pose_noise_comes_modelled_and_not():
    s = scenarios(every=[2], drop=[0.3], pos_sigma=[0.1], yaw_sigma=[0.01], bias_pos=[0.2], bias_yaw=[0.01])
    assert s[0][0] == "baseline" and [x[0] for x in s].count("pose noise") == 4 and {x[4] for x in s if x[0] == "pose noise"} == {True, False}


def test_a_scenario_run_has_all_columns(folder, scans):
    t = run_scenarios(folder, scans, scenarios(every=[2], pos_sigma=[0.1]), repeats=1, burn_in=3)
    assert list(t.columns) == COLUMNS and len(t) == 1 + 1 + 2 and t["hz"].iloc[1] < t["hz"].iloc[0]


def test_a_large_pose_error_makes_the_tracking_worse(folder, scans):
    base = evaluate(folder, scans, burn_in=3)
    bad = evaluate(folder, perturb_scans(scans, pos_sigma=0.5, yaw_sigma=np.radians(3.0)), burn_in=3)
    # with such a pose error the unmodelled noise can make the tracker lose the hazards altogether (no scored rows, NaN errors)
    assert bad["rows"] < base["rows"] or bad["pos_rms"] > base["pos_rms"] or bad["vel_rms_still"] > base["vel_rms_still"]


def test_the_tool_writes_the_summary(folder, tmp_path):
    t = main(["--recording", str(folder), "--every", "2", "--drop", "", "--pos-sigma", "0.1", "--yaw-sigma-deg", "", "--bias-pos", "",
              "--bias-yaw-deg", "1", "--repeats", "1", "--burn-in", "3", "--out", str(tmp_path)])
    assert (tmp_path / "summary.txt").exists() and len(t) == 1 + 1 + 2 + 1