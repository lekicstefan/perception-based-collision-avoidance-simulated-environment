"""The fusion study on the synthetic recording (step 6.6)."""
import numpy as np
import pytest

from testing.support.association_study import scan_measurements
from testing.support.fusion_study import (camera_timeline, leave_latency, replay_fusion, summarise_ablation, summarise_fusion,
                                          summarise_outputs)
from testing.support.synthetic_recording import write_recording
from testing.tools.tracking.fusion_check import main


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_recording(tmp_path_factory.mktemp("rec") / "synthetic")


@pytest.fixture(scope="module")
def scans(folder):
    return scan_measurements(folder, use_camera=True)


def test_the_camera_timeline_has_a_pose_per_frame(folder):
    tl = camera_timeline(folder, every=3)
    assert len(tl) > 5 and all(len(p) == 6 for _, _, p in tl)
    assert np.all(np.diff([t for t, _, _ in tl]) > 0)


def test_without_the_camera_nothing_is_applied(folder, scans):
    r = replay_fusion(folder, scans, use_camera=False, burn_in=3)
    assert r["reasons"] == {} and r["camera_frames"] == 0 and len(r["rows"]) > 0


def test_with_the_camera_bearings_are_applied_and_the_tracks_stay_good(folder, scans):
    off = summarise_fusion(replay_fusion(folder, scans, False, burn_in=3)["rows"])
    run = replay_fusion(folder, scans, True, burn_in=3, camera_every=3)
    on = summarise_fusion(run["rows"])
    assert run["camera_frames"] > 5 and sum(v for k, v in run["reasons"].items() if k in ("ok", "same_time", "out_of_order")) > 0
    car_off, car_on = off[off["id"] == 1].iloc[0], on[on["id"] == 1].iloc[0]
    assert car_on["phantom_rms"] < 0.5 and car_on["cross_rms"] < car_off["cross_rms"] + 0.3


def test_the_tool_writes_its_outputs(folder, tmp_path):
    table = main(["--recording", str(folder), "--out", str(tmp_path), "--camera-every", "3", "--burn-in", "3"])
    assert set(table["mode"]) == {"lidar", "lidar+camera"} and (tmp_path / "summary.txt").exists()


def test_the_output_and_ablation_tables_have_their_columns_and_sane_values(folder, scans):
    rows = replay_fusion(folder, scans, False, burn_in=3)["rows"]
    out, abl = summarise_outputs(rows), summarise_ablation(rows)
    assert {"static_while_moving", "static_while_still", "leave_latency_s"} <= set(out.columns)
    assert {"raw_vel_rms", "raw_over_filter", "raw_static_while_moving"} <= set(abl.columns)
    assert ((out["static_rate"] >= 0) & (out["static_rate"] <= 1)).all() and (abl["raw_vel_rms"] >= 0).all()
    walker = out[out["id"] == 2].iloc[0]
    assert walker["static_while_moving"] == 0.0                                  # a mover is never called static


def test_the_leave_latency_is_the_time_from_the_start_of_motion_to_the_end_of_the_static_flag():
    import pandas as pd
    g = pd.DataFrame(dict(t=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], true_speed=[0, 0, 3, 3, 3, 3], static=[True, True, True, True, False, False]))
    assert leave_latency(g) == pytest.approx(0.2)
    assert pd.isna(leave_latency(g.assign(true_speed=0.0)))