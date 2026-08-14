from pathlib import Path
import hashlib
import json

import pytest

from bearvision.adapters import FfmpegVideoClipper
from bearvision.config.models import ClipExtractionConfig
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
    assert result.assignments[0].selected_user_email == "rider-video@scenario.invalid"
    detected = [entry for entry in result.trace if entry.kind == "person_detected"]
    assert detected[0].at_s == pytest.approx(6.006, abs=0.01)
    assert result.captures
    assert result.uploads[0].object_key == "input-queue/ready/capture-video-frame-180"
    output = tmp_path / "capture-video-frame-180.virtual-cameraman.mp4"
    assert output.is_file()
    extracted = tmp_path / "capture-video-frame-180.mp4"
    assert extracted.is_file()
    tracking = tmp_path / "capture-video-frame-180.tracking.json"
    debug = tmp_path / "capture-video-frame-180.tracking-debug.mp4"
    assert tracking.is_file()
    assert debug.is_file()
    assert output.stat().st_size < extracted.stat().st_size
    assert any(entry.kind == "virtual_cameraman_completed" for entry in result.trace)
    tracking_events = [entry for entry in result.trace if entry.kind == "tracking_observation"]
    assert tracking_events
    assert tracking_events[0].payload["estimate"]
    assert tracking_events[0].payload["camera_center"]
    assert tracking_events[0].payload["confidence_radius_95_px"] > 0
    tracking_metadata = json.loads(tracking.read_text(encoding="utf-8"))
    assert tracking_metadata["tracking_schema_version"] == "2.0"
    assert "rts_smoother" in tracking_metadata["state_estimator"]
    assert tracking_metadata["camera_path"]["zero_phase"] is True
    assert (
        tracking_metadata["length_adjustment"]["output_duration_s"]
        <= tracking_metadata["length_adjustment"]["source_duration_s"]
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    assert FfmpegVideoClipper(ClipExtractionConfig())._probe(output)["has_audio"] is False

    output_capture = cv2.VideoCapture(str(output))
    extracted_capture = cv2.VideoCapture(str(extracted))
    source_capture = cv2.VideoCapture(str(source))
    try:
        output_duration = extracted_capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        ) / extracted_capture.get(
            cv2.CAP_PROP_FPS
        )
        processed_duration = output_capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        ) / output_capture.get(
            cv2.CAP_PROP_FPS
        )
        source_capture.set(cv2.CAP_PROP_POS_MSEC, detected[0].at_s * 1000)
        source_ok, source_frame = source_capture.read()
        extracted_ok, extracted_frame = extracted_capture.read()
        processed_ok, processed_frame = output_capture.read()
        assert source_ok and extracted_ok and processed_ok
        assert output_duration == pytest.approx(5.0, abs=0.1)
        assert processed_duration == pytest.approx(
            tracking_metadata["length_adjustment"]["output_duration_s"], abs=0.1
        )
        mean_absolute_error = abs(
            source_frame.astype("float32") - extracted_frame.astype("float32")
        ).mean()
        assert mean_absolute_error < 12
        assert processed_frame.shape[:2] == (90, 160)
    finally:
        output_capture.release()
        extracted_capture.release()
        source_capture.release()
