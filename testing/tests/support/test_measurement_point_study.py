"""The measurement point study on the synthetic recording (step 6.3)."""
import numpy as np
import pytest

from server.tracking.noise import POINTS
from testing.support.measurement_point_study import StudyConfig, aggregate, collect, spikes, study
from testing.support.synthetic_recording import write_recording
from testing.tools.tracking.measurement_point import main


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_recording(tmp_path_factory.mktemp("rec") / "synthetic")


@pytest.fixture(scope="module")
def samples(folder):
    return collect(folder)


def test_collect_finds_the_hazards_with_their_truth(samples):
    assert {1, 2} <= set(samples["id"])

    walker = samples[samples["id"] == 2]
    assert walker["true_vy"].abs().mean() == pytest.approx(2.0, abs=0.2)


def test_study_has_a_row_per_hazard_and_point(samples):
    series, summary = study(samples, cfg=StudyConfig(burn_in=3, min_samples=5))
    assert set(summary["point"]) == set(POINTS)
    assert set(series["point"]) == set(POINTS)
    assert {"phantom_rms", "vel_rms", "raw_rms", "bias", "offset_sd"} <= set(summary.columns)


def test_a_parked_car_has_a_small_phantom_speed_and_the_walker_is_tracked(samples):
    _, summary = study(samples, cfg=StudyConfig(burn_in=3, min_samples=5))
    car = summary[(summary["id"] == 1)]
    assert car["static"].all()
    assert (car["phantom_rms"] < 0.5).all()
    walker = summary[(summary["id"] == 2)]
    assert not walker["static"].any()
    assert (walker["vel_rms"] < 1.5).all()


def test_the_filter_beats_raw_differencing_on_a_mover(samples):
    _, summary = study(samples, cfg=StudyConfig(burn_in=3, min_samples=5))
    agg = aggregate(summary)
    assert len(agg) == len(POINTS)
    assert (agg["mover_vel_rms"] <= agg["mover_raw_rms"] + 1e-9).all()


def test_the_tool_writes_its_outputs(folder, tmp_path):
    summary = main(["--recording", str(folder), "--out", str(tmp_path), "--burn-in", "3", "--min-samples", "5"])
    assert (tmp_path / "summary.txt").exists() and (tmp_path / "series.csv").exists()
    assert len(summary) > 0


def test_spikes_lists_the_last_measurements_of_every_hazard(samples):
    series, _ = study(samples, cfg=StudyConfig(burn_in=3, min_samples=5))
    sp = spikes(series, "centre", min_error_mps=1e9, last=2)             # no error is that big: only the tails remain
    assert len(sp) == 2 * samples["id"].nunique()
    assert {"range_m", "n_cells", "jump_m", "vel_err"} <= set(sp.columns)