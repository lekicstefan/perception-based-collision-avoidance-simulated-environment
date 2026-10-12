"""The tuning sweeps on the synthetic recording (step 6.9)."""
import numpy as np
import pandas as pd
import pytest

from testing.support.association_study import scan_measurements
from testing.support.synthetic_recording import write_recording
from testing.support.tuning_study import METRICS, metrics, run_one, sweep
from testing.tools.tracking.tuning_check import main


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_recording(tmp_path_factory.mktemp("rec") / "synthetic")


@pytest.fixture(scope="module")
def scans(folder):
    return scan_measurements(folder, use_camera=True)


def test_the_sweep_has_a_default_row_and_one_row_per_value(folder, scans):
    t = sweep(folder, scans, dict(sigma_a=[0.5, 3.0], gate=[6.0]), burn_in=3)
    assert list(t["parameter"]) == ["default", "sigma_a", "sigma_a", "gate"]
    assert list(t.columns[2:]) == METRICS and (t["rows"] > 0).all()


def test_an_unknown_parameter_is_rejected(folder, scans):
    with pytest.raises(ValueError):
        sweep(folder, scans, dict(nonsense=[1.0]))


def test_a_larger_measurement_noise_changes_the_result(folder, scans):
    a = metrics(run_one(folder, scans, dict(sigma_extent=0.2), burn_in=3))
    b = metrics(run_one(folder, scans, dict(sigma_extent=2.0), burn_in=3))
    assert a["rows"] > 0 and b["rows"] > 0 and (a["vel_rms_still"] != b["vel_rms_still"] or a["pos_rms"] != b["pos_rms"])


def test_sigma_px_runs_with_the_camera(folder, scans):
    r = run_one(folder, scans, dict(sigma_px=2.0), camera_every=3, burn_in=3)
    assert r["camera_frames"] > 5


def test_empty_rows_give_nan_metrics():
    m = metrics(dict(rows=pd.DataFrame(), nis_mean=float("nan")))
    assert m["rows"] == 0 and np.isnan(m["pos_rms"])


def test_the_tool_writes_the_summary(folder, tmp_path):
    t = main(["--recording", str(folder), "--sigma-a", "1,2", "--sigma-extent", "", "--gate", "", "--burn-in", "3", "--out", str(tmp_path)])
    assert (tmp_path / "summary.txt").exists() and len(t) == 3