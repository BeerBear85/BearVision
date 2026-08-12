"""End-to-end regression over Blender video and generated BearTag samples."""

from pathlib import Path

import pytest

from bearvision.contracts import load_scenario
from bearvision.edge import build_behavioral_system


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "code/dnn_models/yolov8n.onnx"
SCENARIO = ROOT / "specs/scenarios/wakeboard-fs360-60fps-blender-regression.yaml"


def test_blender_video_and_synthetic_bear_tag_assign_the_rider(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    if not MODEL.is_file() or MODEL.stat().st_size < 1_000_000:
        pytest.skip("YOLO Git LFS asset is not materialized")
    scenario = load_scenario(SCENARIO)
    assert scenario.video is not None
    if not (ROOT / scenario.video.path).is_file():
        pytest.skip("local ignored Blender MP4 is not available")

    result = build_behavioral_system(scenario, capture_dir=tmp_path).run()

    assert result.failures == ()
    assert result.expectation_failures == ()
    assert result.assignments[0].rider_id == "rider-wakeboard-fs360-60fps"
    evidence = result.assignments[0].evidence[0]
    assert evidence.observation_count >= 30
    assert evidence.median_rssi_dbm == pytest.approx(-79)
    assert evidence.qualifies
    detected = [entry for entry in result.trace if entry.kind == "person_detected"]
    assert detected[0].at_s == pytest.approx(1.8, abs=0.11)
    assert result.uploads[0].object_key.startswith("rider-wakeboard-fs360-60fps/")
    assert (tmp_path / "capture-video-frame-108.mp4").is_file()
