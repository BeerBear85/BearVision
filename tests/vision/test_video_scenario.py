from pathlib import Path

import pytest


pytest.importorskip("cv2")

from bearvision.contracts import load_scenario
from bearvision.edge import build_behavioral_system


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "code/dnn_models/yolov8n.onnx"


def test_recorded_video_drives_real_yolo_capture_and_rider_assignment() -> None:
    if not MODEL.is_file() or MODEL.stat().st_size < 1_000_000:
        pytest.skip("YOLO Git LFS asset is not materialized")
    scenario = load_scenario(ROOT / "specs/scenarios/wakeboard-video-yolo.yaml")

    result = build_behavioral_system(scenario).run()

    assert result.failures == ()
    assert result.expectation_failures == ()
    assert result.assignments[0].rider_id == "rider-video"
    detected = [entry for entry in result.trace if entry.kind == "person_detected"]
    assert detected[0].at_s == pytest.approx(6.006, abs=0.01)
    assert result.captures
    assert result.uploads[0].object_key == "rider-video/preview_low.mp4"
