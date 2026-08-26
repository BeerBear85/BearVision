"""End-to-end regression over Blender video and generated BearTag samples."""

import json
from pathlib import Path

import pytest

from bearvision.config import load_edge_config
from bearvision.contracts import load_scenario
from bearvision.edge import build_behavioral_system

cv2 = pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "code/dnn_models/yolov8n.onnx"
EDGE_CONFIG = load_edge_config(ROOT / "config/edge.yaml")
SCENARIO = ROOT / "specs/scenarios/wakeboard-fs360-60fps-blender-regression.yaml"
TWO_RIDER_SCENARIO = (
    ROOT / "specs/scenarios/wakeboard-two-riders-60fps-blender-regression.yaml"
)


def test_blender_video_and_synthetic_bear_tag_assign_the_rider(tmp_path: Path) -> None:
    if not MODEL.is_file() or MODEL.stat().st_size < 1_000_000:
        pytest.skip("YOLO Git LFS asset is not materialized")
    scenario = load_scenario(SCENARIO)
    assert scenario.video is not None
    if not (ROOT / scenario.video.path).is_file():
        pytest.skip("local ignored Blender MP4 is not available")

    result = build_behavioral_system(
        scenario,
        edge_config=EDGE_CONFIG,
        capture_dir=tmp_path,
    ).run()

    assert result.failures == ()
    assert result.expectation_failures == ()
    assert result.assignments[0].selected_bear_tag_id == "bear_tag_666"
    assert result.assignments[0].selected_user_id is not None
    evidence = result.assignments[0].candidates[0]
    assert evidence.observation_count >= 30
    assert evidence.median_rssi_dbm == pytest.approx(-80)
    assert evidence.qualifies
    detected = [entry for entry in result.trace if entry.kind == "person_detected"]
    assert detected[0].at_s == pytest.approx(2.1, abs=0.11)
    assert result.uploads[0].object_key == "input-queue/ready/capture-video-frame-126"
    capture_event = next(
        entry for entry in result.trace if entry.kind == "capture_completed"
    )
    processing_event = next(
        entry for entry in result.trace if entry.kind == "virtual_cameraman_completed"
    )
    source_clip = tmp_path / capture_event.payload["filename"]
    processed_clip = tmp_path / processing_event.payload["processed_filename"]
    tracking_path = tmp_path / processing_event.payload["tracking_filename"]
    assert source_clip.name == "capture-video-frame-126-GX010001.MP4"
    assert (
        tmp_path / ".simulated-gopro-sd/100GOPRO/GX010001.MP4"
    ).is_file()
    assert source_clip.is_file()
    assert processed_clip.is_file()
    assert processed_clip.stat().st_size > 0

    tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
    selected_measurements = [
        frame for frame in tracking["frames"] if frame["detection"] is not None
    ]
    assert len(selected_measurements) >= 15
    assert selected_measurements[-1]["estimate"]["x_px"] > 1_800
    assert tracking["length_adjustment"]["adjusted"] is True
    assert tracking["length_adjustment"]["output_duration_s"] < 5.0

    capture = cv2.VideoCapture(str(processed_clip))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
        ok, frame = capture.read()
        assert ok
        assert frame.shape[:2] == (
            EDGE_CONFIG.virtual_cameraman.output_height_px,
            EDGE_CONFIG.virtual_cameraman.output_width_px,
        )
    finally:
        capture.release()


def test_two_rider_blender_video_assigns_both_riders_bear_tags(
    tmp_path: Path,
) -> None:
    if not MODEL.is_file() or MODEL.stat().st_size < 1_000_000:
        pytest.skip("YOLO Git LFS asset is not materialized")
    scenario = load_scenario(TWO_RIDER_SCENARIO)
    assert scenario.video is not None
    if not (ROOT / scenario.video.path).is_file():
        pytest.skip("local ignored Blender MP4 is not available")
    assert [series.tag_id for series in scenario.synthetic_bear_tags] == [
        "bear_tag_666",
        "bear_tag_123",
    ]

    result = build_behavioral_system(
        scenario,
        edge_config=EDGE_CONFIG,
        capture_dir=tmp_path,
    ).run()

    assert result.failures == ()
    assert result.expectation_failures == ()
    assert [assignment.selected_bear_tag_id for assignment in result.assignments] == [
        "bear_tag_666",
        "bear_tag_123",
    ]
    assert all(assignment.selected_user_id is not None for assignment in result.assignments)
    assert result.assignments[0].selected_user_id != result.assignments[1].selected_user_id
    detected = [entry for entry in result.trace if entry.kind == "person_detected"]
    assert detected[0].at_s == pytest.approx(2.1, abs=0.11)
    assert any(entry.at_s == pytest.approx(11.8, abs=0.11) for entry in detected)
    assert [upload.object_key for upload in result.uploads] == [
        "input-queue/ready/capture-video-frame-126",
        "input-queue/ready/capture-video-frame-708",
    ]
    capture_events = [
        entry for entry in result.trace if entry.kind == "capture_completed"
    ]
    processing_events = [
        entry for entry in result.trace if entry.kind == "virtual_cameraman_completed"
    ]
    assert [entry.payload["filename"] for entry in capture_events] == [
        "capture-video-frame-126-GX010001.MP4",
        "capture-video-frame-708-GX010002.MP4",
    ]
    assert [path.name for path in sorted(
        (tmp_path / ".simulated-gopro-sd/100GOPRO").glob("*.MP4")
    )] == ["GX010001.MP4", "GX010002.MP4"]
    for event in processing_events:
        capture = cv2.VideoCapture(str(tmp_path / event.payload["processed_filename"]))
        try:
            assert capture.isOpened()
            ok, frame = capture.read()
            assert ok
            assert frame.shape[:2] == (
                EDGE_CONFIG.virtual_cameraman.output_height_px,
                EDGE_CONFIG.virtual_cameraman.output_width_px,
            )
        finally:
            capture.release()
