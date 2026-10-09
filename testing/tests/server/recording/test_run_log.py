"""Tests for the Python run logging."""
import csv
import json

from server.recording.run_log import RunLog, TableLog, git_info, package_versions


def test_table_log_writes_header_and_rows(tmp_path):
    t = TableLog(tmp_path / "t.csv", ["a", "b"])
    t.row(1, 2.5)
    t.row(2, "")
    t.close()
    rows = list(csv.reader(open(tmp_path / "t.csv", encoding="utf-8")))
    assert rows == [["a", "b"], ["1", "2.5"], ["2", ""]]


def test_run_log_writes_meta_summary_and_tables(tmp_path):
    log = RunLog(tmp_path / "run1")
    log.write_meta(rate_hz=30.0)
    log.write_summary(cycles=3)
    log.table("cycles", ["cycle"]).row(0)
    log.close()
    meta = json.loads((tmp_path / "run1" / "python" / "meta.json").read_text(encoding="utf-8"))
    assert meta["rate_hz"] == 30.0 and "numpy" in meta["packages"] and "commit" in meta["git"]
    assert json.loads((tmp_path / "run1" / "python" / "summary.json").read_text(encoding="utf-8"))["cycles"] == 3
    assert (tmp_path / "run1" / "python" / "cycles.csv").exists()


def test_git_info_and_versions_never_raise():
    assert set(git_info()) == {"commit", "branch", "dirty"}
    assert isinstance(package_versions(), dict)