"""End-to-end regression over Blender video and generated BearTag samples."""

import json
from pathlib import Path

import pytest

from bearvision.contracts import load_scenario
from bearvision.edge import build_behavioral_system

cv2 = pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "code/dnn_models/yolov8n.onnx"
SCENARIO = ROOT / "specs/scenarios/wakeboard-fs360-60fps-blender-regression.yaml"


def test_blender_video_and_synthetic_bear_tag_assign_the_rider(tmp_path: Path) -> None:
    if not MODEL.is_file() or MODEL.stat().st_size < 1_000_000:
        pytest.skip("YOLO Git LFS asset is not materialized")
    scenario = load_scenario(SCENARIO)
    assert scenario.video is not None
    if not (ROOT / scenario.video.path).is_file():
        pytest.skip("local ignored Blender MP4 is not available")

    result = build_behavioral_system(scenario, capture_dir=tmp_path).run()

    assert result.failures == ()
    assert result.expectation_failures == ()
    assert (
        result.assignments[0].selected_user_email
        == "rider-wakeboard-fs360-60fps@scenario.invalid"
    )
    evidence = result.assignments[0].evidence[0]
    assert evidence.observation_count >= 30
    assert evidence.median_rssi_dbm == pytest.approx(-79)
    assert evidence.qualifies
    detected = [entry for entry in result.trace if entry.kind == "person_detected"]
    assert detected[0].at_s == pytest.approx(1.8, abs=0.11)
    assert result.uploads[0].object_key == "input-queue/ready/capture-video-frame-108"
    source_clip = tmp_path / "capture-video-frame-108.mp4"
    processed_clip = tmp_path / "capture-video-frame-108.virtual-cameraman.mp4"
    tracking_path = tmp_path / "capture-video-frame-108.tracking.json"
    assert source_clip.is_file()
    assert processed_clip.is_file()
    assert processed_clip.stat().st_size > 0

    tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
    selected_measurements = [
        frame for frame in tracking["frames"] if frame["detection"] is not None
    ]
    assert len(selected_measurements) >= 20
    assert selected_measurements[-1]["estimate"]["x_px"] > 1_800
    assert tracking["length_adjustment"]["adjusted"] is True
    assert tracking["length_adjustment"]["output_duration_s"] < 5.0

    capture = cv2.VideoCapture(str(processed_clip))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
        ok, frame = capture.read()
        assert ok
        assert frame.shape[:2] == (90, 160)
    finally:
        capture.release()
