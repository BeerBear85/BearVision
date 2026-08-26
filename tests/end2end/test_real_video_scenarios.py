from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math

import pytest

from bearvision.adapters import FfmpegVideoClipper
from bearvision.config import load_edge_config
from bearvision.config.models import ClipExtractionConfig
from bearvision.contracts import ScenarioDefinition, load_scenario
from bearvision.edge import build_behavioral_system
from bearvision.server import FileSystemJobQueue


cv2 = pytest.importorskip("cv2")
pytestmark = pytest.mark.end2end


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "code/dnn_models/yolov8n.onnx"
EDGE_CONFIG = load_edge_config(ROOT / "config/edge.yaml")


@dataclass(frozen=True, slots=True)
class RealVideoCase:
    scenario_filename: str
    video_path: str
    selected_tag_ids: tuple[str, ...]


CASES = (
    RealVideoCase(
        "wakeboard-video-yolo.yaml",
        "tests/data/preview_low.mp4",
        ("bear_tag_666",),
    ),
    RealVideoCase(
        "wakeboard-testmovie1-yolo.yaml",
        "tests/data/TestMovie1.mp4",
        ("bear_tag_666",),
    ),
    RealVideoCase(
        "wakeboard-testmovie3-yolo.yaml",
        "tests/end2end/input_video/TestMovie3.avi",
        ("bear_tag_666", "bear_tag_666"),
    ),
    RealVideoCase(
        "wakeboard-testmovie5-two-riders-yolo.yaml",
        "tests/data/TestMovie5_two_persons.mp4",
        ("bear_tag_666", "bear_tag_667"),
    ),
)


def scenario_for(case: RealVideoCase) -> ScenarioDefinition:
    return load_scenario(ROOT / "specs/scenarios" / case.scenario_filename)


@pytest.mark.parametrize("case", CASES, ids=lambda case: Path(case.video_path).stem)
def test_real_video_scenario_declares_estimated_telemetry_for_every_rider(
    case: RealVideoCase,
) -> None:
    scenario = scenario_for(case)

    assert scenario.video is not None
    assert scenario.video.path == case.video_path
    assert (ROOT / case.video_path).is_file()
    assert scenario.components.frames == "video"
    assert scenario.components.detector == "yolo"
    assert scenario.components.bear_tag == "synthetic"
    assert scenario.components.camera == "simulated_gopro"
    assert scenario.synthetic_bear_tags
    assert scenario.synthetic_bear_tags[0].tag_id == "bear_tag_666"
    assert len({series.tag_id for series in scenario.synthetic_bear_tags}) == len(
        scenario.synthetic_bear_tags
    )

    for series in scenario.synthetic_bear_tags:
        baseline = series.baseline_acceleration_mps2
        baseline_norm = math.sqrt(baseline.x**2 + baseline.y**2 + baseline.z**2)
        assert baseline_norm == pytest.approx(9.80665, abs=0.01)
        assert series.motion_windows
        for window in series.motion_windows:
            acceleration = window.acceleration_mps2
            active_norm = math.sqrt(
                acceleration.x**2 + acceleration.y**2 + acceleration.z**2
            )
            assert active_norm - baseline_norm >= 2.0
            assert window.rssi_dbm is not None
            assert window.rssi_dbm > series.rssi_dbm


@pytest.mark.parametrize("case", CASES, ids=lambda case: Path(case.video_path).stem)
def test_real_video_runs_through_yolo_virtual_camera_queue_and_assignment(
    case: RealVideoCase,
    tmp_path: Path,
) -> None:
    if not MODEL.is_file() or MODEL.stat().st_size < 1_000_000:
        pytest.skip("YOLO Git LFS asset is not materialized")
    scenario = scenario_for(case)
    assert scenario.video is not None
    source = ROOT / scenario.video.path
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    capture_dir = tmp_path / "captures"
    queue = FileSystemJobQueue(tmp_path / "shared-queue")

    result = build_behavioral_system(
        scenario,
        edge_config=EDGE_CONFIG,
        capture_dir=capture_dir,
        job_queue=queue,
        process_server=True,
    ).run()

    assert result.failures == ()
    assert result.expectation_failures == ()
    assert len(result.captures) == len(case.selected_tag_ids)
    assert len(result.uploads) == len(case.selected_tag_ids)
    assert tuple(item.selected_bear_tag_id for item in result.assignments) == (
        case.selected_tag_ids
    )
    assert all(item.status == "processed" for item in result.assignments)
    assert queue.snapshot()["counts"] == {
        "ready": 0,
        "processing": 0,
        "processed": len(case.selected_tag_ids),
        "unresolved": 0,
        "failed": 0,
    }

    detected = [entry for entry in result.trace if entry.kind == "person_detected"]
    assert detected
    assert scenario.expect.first_detection_between_s is not None
    first_start_s, first_end_s = scenario.expect.first_detection_between_s
    assert first_start_s <= detected[0].at_s <= first_end_s

    completed = [entry for entry in result.trace if entry.kind == "capture_completed"]
    virtual_camera = [
        entry for entry in result.trace if entry.kind == "virtual_cameraman_completed"
    ]
    assert len(completed) == len(case.selected_tag_ids)
    assert len(virtual_camera) == len(case.selected_tag_ids)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash

    clipper = FfmpegVideoClipper(ClipExtractionConfig())
    for capture_event, processed_event in zip(completed, virtual_camera, strict=True):
        extracted = capture_dir / capture_event.payload["filename"]
        output = capture_dir / processed_event.payload["processed_filename"]
        tracking = capture_dir / processed_event.payload["tracking_filename"]
        debug = capture_dir / processed_event.payload["debug_video_filename"]
        assert extracted.is_file() and extracted.stat().st_size > 0
        assert output.is_file() and output.stat().st_size > 0
        assert tracking.is_file() and tracking.stat().st_size > 0
        assert debug.is_file() and debug.stat().st_size > 0
        assert clipper._probe(output)["has_audio"] is False

        tracking_metadata = json.loads(tracking.read_text(encoding="utf-8"))
        assert tracking_metadata["tracking_schema_version"] == "2.0"
        assert "rts_smoother" in tracking_metadata["state_estimator"]
        assert tracking_metadata["camera_path"]["zero_phase"] is True
        assert (
            tracking_metadata["length_adjustment"]["output_duration_s"]
            <= tracking_metadata["length_adjustment"]["source_duration_s"]
        )

        source_capture = cv2.VideoCapture(str(source))
        extracted_capture = cv2.VideoCapture(str(extracted))
        output_capture = cv2.VideoCapture(str(output))
        try:
            source_capture.set(
                cv2.CAP_PROP_POS_MSEC,
                float(capture_event.payload["clip_start_s"]) * 1000,
            )
            source_ok, source_frame = source_capture.read()
            extracted_ok, extracted_frame = extracted_capture.read()
            output_ok, output_frame = output_capture.read()
            assert source_ok and extracted_ok and output_ok
            mean_absolute_error = abs(
                source_frame.astype("float32") - extracted_frame.astype("float32")
            ).mean()
            assert mean_absolute_error < 12
            assert output_frame.shape[:2] == (
                EDGE_CONFIG.virtual_cameraman.output_height_px,
                EDGE_CONFIG.virtual_cameraman.output_width_px,
            )
        finally:
            source_capture.release()
            extracted_capture.release()
            output_capture.release()
