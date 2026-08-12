from pathlib import Path
import hashlib

import pytest

from bearvision.contracts import load_scenario
from bearvision.edge import build_behavioral_system

cv2 = pytest.importorskip("cv2")


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "code/dnn_models/yolov8n.onnx"


def test_recorded_video_drives_real_yolo_capture_and_rider_assignment(
    tmp_path: Path,
) -> None:
    if not MODEL.is_file() or MODEL.stat().st_size < 1_000_000:
        pytest.skip("YOLO Git LFS asset is not materialized")
    scenario = load_scenario(ROOT / "specs/scenarios/wakeboard-video-yolo.yaml")
    source = ROOT / scenario.video.path  # type: ignore[union-attr]
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    result = build_behavioral_system(scenario, capture_dir=tmp_path).run()

    assert result.failures == ()
    assert result.expectation_failures == ()
    assert result.assignments[0].rider_id == "rider-video"
    detected = [entry for entry in result.trace if entry.kind == "person_detected"]
    assert detected[0].at_s == pytest.approx(6.006, abs=0.01)
    assert result.captures
    assert result.uploads[0].object_key == "rider-video/capture-video-frame-180.mp4"
    output = tmp_path / "capture-video-frame-180.mp4"
    assert output.is_file()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash

    output_capture = cv2.VideoCapture(str(output))
    source_capture = cv2.VideoCapture(str(source))
    try:
        output_duration = output_capture.get(cv2.CAP_PROP_FRAME_COUNT) / output_capture.get(
            cv2.CAP_PROP_FPS
        )
        source_capture.set(cv2.CAP_PROP_POS_MSEC, detected[0].at_s * 1000)
        source_ok, source_frame = source_capture.read()
        output_ok, output_frame = output_capture.read()
        assert source_ok and output_ok
        assert output_duration == pytest.approx(5.0, abs=0.1)
        mean_absolute_error = abs(
            source_frame.astype("float32") - output_frame.astype("float32")
        ).mean()
        assert mean_absolute_error < 12
    finally:
        output_capture.release()
        source_capture.release()
