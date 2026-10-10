"""The evaluation end to end on a synthetic recording (testing/support/synthetic_recording.py): files, run and tool."""
import pytest

from server.perception.camera_refinement import RefinementConfig
from server.perception.ground import GroundConfig, GroundMethod
from server.perception.pipeline import PerceptionConfig
from testing.support.evaluation_run import evaluate_recording
from testing.support.ground_truth import GroundTruth
from testing.support.perception_eval import EvalConfig, summarise
from testing.support.synthetic_recording import write_recording
from testing.tools.perception import evaluate


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    return write_recording(tmp_path_factory.mktemp("rec") / "synthetic")


def test_the_synthetic_recording_has_all_the_files_and_truth(recording):
    assert GroundTruth.exists(recording)
    objs = GroundTruth.load(recording).objects_at(0.5)
    assert {o.label for o in objs} == {"car", "person"} and len(objs) == 3
    far = next(o for o in objs if o.centre[0] > 70)
    assert far.hits_returned is not None and far.hits_returned < 3               # not a detectable target


@pytest.mark.parametrize("method", list(GroundMethod))
@pytest.mark.parametrize("camera", [False, True])
def test_everything_detectable_is_found_with_either_ground_method_and_with_or_without_the_camera(recording, method, camera):
    cfg = PerceptionConfig(ground=GroundConfig(method=method), refinement=RefinementConfig(enabled=camera))
    haz, cl = evaluate_recording(recording, cfg, EvalConfig(min_hits=2), use_camera=camera)
    s = summarise(haz, cl)
    assert s["detectable_hazard_scans"] > 20
    assert s["recall"] == 1.0 and s["precision"] == 1.0 and s["fragmented_fraction"] == 0.0
    assert s["nearest_err"]["max_abs"] < 0.05                                      # the nearest point is on the surface
    assert s["centre_err"]["median"] < s["centroid_err"]["median"] + 0.2


def test_a_hazard_with_too_few_cells_is_not_counted_against_the_recall(recording):
    cfg = PerceptionConfig()
    haz, _ = evaluate_recording(recording, cfg, EvalConfig(min_hits=50))             # nobody has that many cells at 78 m or so
    assert haz[haz["id"] == 3]["detectable"].sum() == 0


def test_the_tool_writes_summary_csv_and_plots(recording, tmp_path):
    out = tmp_path / "out"
    results = evaluate.main(["--recording", str(recording), "--compare", "--out", str(out), "--scans", "0:12:3"])
    assert set(results) == {"height", "height+camera", "range_image", "range_image+camera"}
    for name in results:
        assert (out / name / "hazards.csv").exists() and (out / name / "summary.json").exists()
    for f in ("summary.txt", "recall_vs_range.png", "recall_vs_hits.png", "errors_vs_range.png", "timeline.png"):
        assert (out / f).stat().st_size > 1000