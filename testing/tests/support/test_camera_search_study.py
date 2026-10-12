"""Camera edge search on the synthetic recording (step 6.9)."""
from dataclasses import replace

import numpy as np
import pytest

from server.tracking.bearing import BearingConfig
from testing.support.association_study import scan_measurements
from testing.support.camera_search_study import (DEFAULT, METRICS, SPACE, TUNED, Evaluator, build_bearing, clip, edge_metrics, perturb, sample,
                                                  score, search, sensitivity, silhouette)
from testing.support.synthetic_recording import write_recording
from testing.tools.tracking.camera_search_check import main


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_recording(tmp_path_factory.mktemp("rec") / "synthetic")


@pytest.fixture(scope="module")
def scans(folder):
    return scan_measurements(folder, use_camera=True)


def test_the_tuned_set_is_the_default_bearing_config():
    b, d = build_bearing(TUNED), BearingConfig()
    assert (b.margin_frac, b.min_margin_px, b.min_width_px, b.width_ratio, b.use_color, b.color_canny_factor) == \
           (d.margin_frac, d.min_margin_px, d.min_width_px, d.width_ratio, d.use_color, d.color_canny_factor)
    assert b.edges.canny_high == pytest.approx(d.edges.canny_high)                          # 41 * (188 / 41) is not exactly 188
    assert replace(b.edges, canny_high=d.edges.canny_high) == d.edges


def test_samples_and_perturbations_stay_in_the_space():
    rng = np.random.default_rng(0)
    for p in sample(rng, 30) + [perturb(rng, DEFAULT, 3.0) for _ in range(20)]:
        for k, (lo, hi, kind) in SPACE.items():
            assert lo <= p[k] <= hi
            if kind == "odd":
                assert p[k] % 2 == 1
    assert clip(dict(DEFAULT, canny_low=500.0))["canny_low"] == SPACE["canny_low"][1]


def test_the_silhouette_is_the_pixel_extent_of_the_box():
    class Cam:
        def project(self, pts):
            return 320 + 400 * pts[:, 1] / pts[:, 0] * -1, 240 - 400 * pts[:, 2] / pts[:, 0], pts[:, 0]

    class Obj:
        height = 1.0

        def footprint(self):
            return np.array([[10.0, 1.0], [10.0, -1.0], [12.0, -1.0], [12.0, 1.0]])

    lo, hi = silhouette(Obj(), Cam())
    assert lo == pytest.approx(320 - 400 * 1.0 / 10.0) and hi == pytest.approx(320 + 400 * 1.0 / 10.0)

    class Behind(Obj):
        def footprint(self):
            return np.array([[-1.0, 1.0], [-1.0, -1.0], [2.0, -1.0], [2.0, 1.0]])

    assert silhouette(Behind(), Cam()) is None


def test_the_score_prefers_good_measurements_and_punishes_false_edges():
    base = dict(good_rate=0.5, wrong_rate=0.0, rms_mid_px=2.0, static_while_moving_max=0.0)
    assert score(dict(base, good_rate=0.6)) < score(base) and score(dict(base, wrong_rate=0.1)) > score(base)
    assert score(dict(base, static_while_moving_max=0.1)) > score(base) and score(dict(base, rms_mid_px=float("nan"))) < score(dict(base, rms_mid_px=10))


def test_edge_metrics_have_the_rates_in_range(folder, scans):
    m = edge_metrics(folder, scans, build_bearing(DEFAULT), camera_every=3, burn_in=3)
    assert set(METRICS) <= set(m) and m["attempts"] >= 0
    for k in ("found_rate", "good_rate", "wrong_rate"):
        assert np.isnan(m[k]) or 0.0 <= m[k] <= 1.0
    assert np.isnan(m["good_rate"]) or m["good_rate"] <= m["found_rate"] + 1e-9


def test_a_small_search_runs(folder, scans):
    ev = Evaluator([(folder, scans)], workers=1, camera_every=6, burn_in=3)
    t = search(ev, 2, 1, 1, 1, seed=0, log=lambda *_: None)
    assert len(t) == 1 + 2 + 1 and t["score"].is_monotonic_increasing and len(sensitivity(t)) == len(SPACE)


def test_the_tool_writes_the_summary(folder, tmp_path):
    t = main(["--recording", str(folder), "--holdout", str(folder), "--random", "1", "--rounds", "1", "--per-round", "1", "--top", "1",
              "--workers", "1", "--camera-every", "6", "--burn-in", "3", "--out", str(tmp_path)])
    assert (tmp_path / "results.csv").exists() and "holdout" in (tmp_path / "summary.txt").read_text() and len(t) == 1 + 1 + 1