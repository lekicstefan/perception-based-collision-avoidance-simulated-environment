"""Candidate scoring on the synthetic recording (step 6.9)."""
import pytest

from testing.support.synthetic_recording import write_recording
from testing.tools.tracking.candidate_check import main, parse_candidates


def test_candidates_are_parsed_on_top_of_the_default():
    c = parse_candidates("a:sigma_a=2,gate=15;b:known_sigma=1")
    assert list(c) == ["default", "a", "b"] and c["a"]["sigma_a"] == 2.0 and c["a"]["sigma_extent"] == c["default"]["sigma_extent"]
    with pytest.raises(ValueError):
        parse_candidates("x:nonsense=1")


def test_the_tool_scores_every_candidate_on_every_recording(tmp_path):
    folder = write_recording(tmp_path / "rec")
    t = main(["--recording", str(folder), "--candidates", "a:sigma_a=2;b:gate=15", "--burn-in", "3", "--out", str(tmp_path / "out")])
    assert list(t["candidate"]) == ["default", "a", "b"] and (tmp_path / "out" / "summary.txt").exists()