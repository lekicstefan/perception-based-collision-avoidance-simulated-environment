"""The tracker study on the synthetic recording (steps 6.4, 6.5)."""
import pytest

from testing.support.association_study import replay, scan_measurements
from testing.support.synthetic_recording import write_recording
from testing.tools.tracking.association_check import main


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_recording(tmp_path_factory.mktemp("rec") / "synthetic")


@pytest.fixture(scope="module")
def scans(folder):
    return scan_measurements(folder)


def test_scan_measurements_carry_the_hazard_ids(scans):
    assert {1, 2} <= {hid for _, items in scans for _, hid in items}


def test_the_hazards_are_followed_without_identity_switches(scans):
    s = replay(scans, 9.21)
    assert s["switches"] == 0
    assert s["correct"] > 0.8 * s["measurements"]
    assert s["confirmed"] >= 2 and s["coverage"] > 0.5


def test_a_gate_below_zero_gates_everything_and_every_measurement_starts_a_track(scans):
    s = replay(scans, -1.0)             # the synthetic scans have no noise: a static hazard repeats its position, d2 = 0 is inside gate 0
    assert s["correct"] == 0 and s["born"] == s["measurements"] and s["confirmed"] == 0


def test_the_tool_writes_its_outputs(folder, tmp_path):
    table = main(["--recording", str(folder), "--out", str(tmp_path), "--gates", "9.21,20"])
    assert len(table) == 2 and (tmp_path / "summary.txt").exists() and (tmp_path / "events.txt").exists()