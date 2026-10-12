"""The parameter search on the synthetic recording (step 6.9, long run)."""
import numpy as np
import pytest

from testing.support.association_study import scan_measurements
from testing.support.search_study import DEFAULT, SPACE, Evaluator, Weights, clip, perturb, sample, score, search, sensitivity
from testing.support.synthetic_recording import write_recording
from testing.tools.tracking.search_check import main


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_recording(tmp_path_factory.mktemp("rec") / "synthetic")


def test_samples_stay_in_the_space_and_are_stratified():
    ps = sample(np.random.default_rng(1), 50)
    for k, (lo, hi, integer) in SPACE.items():
        v = np.array([p[k] for p in ps])
        assert v.min() >= lo and v.max() <= hi
        if not integer:
            assert len(np.unique(np.floor((v - lo) / (hi - lo) * 50))) >= 45          # one per stratum
    assert all(isinstance(p["static_confirm_scans"], int) for p in ps)


def test_perturbing_and_clipping_respect_the_bounds():
    p = perturb(np.random.default_rng(2), DEFAULT, 5.0)
    assert all(SPACE[k][0] <= p[k] <= SPACE[k][1] for k in SPACE)
    assert clip(dict(DEFAULT, sigma_a=100.0))["sigma_a"] == SPACE["sigma_a"][1]


def test_the_score_prefers_the_safe_configuration():
    base = dict(rows=100, vel_rms_moving=1.0, vel_rms_still=0.2, static_while_moving_max=0.0, static_while_still_mean=0.8,
                leave_latency_max_s=0.0)
    assert score(dict(base, static_while_moving_max=0.1), 100) > score(base, 100)
    assert score(dict(base, static_while_still_mean=0.0), 100) > score(base, 100)          # never flagging anything is not a win
    assert score(dict(base, rows=80), 100) > score(base, 100)
    assert score(dict(base, static_while_moving_max=float("nan")), 100) == score(base, 100)


def test_a_small_search_runs_and_the_default_is_in_it(folder):
    ev = Evaluator([(folder, scan_measurements(folder, use_camera=True))], workers=1, burn_in=3)
    t = search(ev, 3, 1, 2, 2, seed=0, log=lambda *_: None)
    assert len(t) == 1 + 3 + 2 * 2 and t["score"].is_monotonic_increasing and (t["stage"] == "random").sum() == 4
    assert len(sensitivity(t)) == len(SPACE)


def test_the_tool_writes_summary_and_results(folder, tmp_path):
    t = main(["--recording", str(folder), "--holdout", str(folder), "--random", "2", "--rounds", "1", "--per-round", "1", "--top", "1",
              "--workers", "1", "--burn-in", "3", "--out", str(tmp_path)])
    assert (tmp_path / "results.csv").exists() and "holdout" in (tmp_path / "summary.txt").read_text() and len(t) == 1 + 2 + 1