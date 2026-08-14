from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from bearvision.contracts import (
    CaptureRequest,
    EdgeJobManifest,
    JobResultManifest,
    JobVideo,
    RiderAssignment,
    RiderAssignmentStatus,
    TagObservation,
    Vector3,
)


def test_tag_observation_is_strict_and_unit_bounded() -> None:
    observation = TagObservation(
        tag_id="tag-17",
        observed_at_utc=datetime(2026, 8, 12, tzinfo=timezone.utc),
        observed_at_monotonic_s=12.5,
        rssi_dbm=-52,
        acceleration_mps2=Vector3(x=0, y=1.2, z=9.81),
        battery_voltage_mv=3000,
    )

    assert observation.contract_schema_version == "2.0"
    assert observation.battery_voltage_mv == 3000
    assert observation.rssi_dbm == -52

    with pytest.raises(ValidationError):
        TagObservation(
            tag_id="tag-17",
            observed_at_utc=datetime(2026, 8, 12),
            observed_at_monotonic_s=12.5,
            rssi_dbm=-200,
            acceleration_mps2=Vector3(x=0, y=0, z=0),
            unexpected=True,
        )


def test_ble_assignment_never_hides_ambiguity() -> None:
    ambiguous = RiderAssignment(
        status=RiderAssignmentStatus.AMBIGUOUS,
        assigned_at_monotonic_s=8,
        candidate_tag_ids=("tag-17", "tag-22"),
        reason="multiple registered tags qualify",
    )
    assert ambiguous.rider_id is None

    with pytest.raises(ValidationError):
        RiderAssignment(
            status=RiderAssignmentStatus.AMBIGUOUS,
            assigned_at_monotonic_s=8,
            rider_id="rider-17",
            tag_id="tag-17",
            reason="invalid hidden assignment",
        )


def test_contracts_generate_json_schema() -> None:
    schema = TagObservation.model_json_schema()
    assert schema["properties"]["contract_schema_version"]["const"] == "2.0"
    assert "rssi_dbm" in schema["properties"]


def test_capture_can_start_before_rider_assignment_is_known() -> None:
    request = CaptureRequest(
        request_id="capture-frame-1",
        requested_at_monotonic_s=4,
        pre_roll_s=0,
        post_roll_s=5,
    )
    assert request.assignment is None


def test_edge_job_rejects_non_utc_time_unsafe_filename_and_non_video_media() -> None:
    utc = datetime(2026, 8, 12, tzinfo=timezone.utc)
    video = {
        "filename": "clip.mp4",
        "mimeType": "video/mp4",
        "sizeBytes": 1,
        "sha256": "0" * 64,
    }
    payload = {
        "jobId": "job-1",
        "edgeDeviceId": "edge-1",
        "createdAt": utc + timedelta(seconds=2),
        "captureStartedAt": utc,
        "captureEndedAt": utc + timedelta(seconds=1),
        "video": video,
    }

    with pytest.raises(ValidationError, match="datetime must be UTC"):
        EdgeJobManifest.model_validate(
            {**payload, "createdAt": datetime(2026, 8, 12, 4, tzinfo=timezone(timedelta(hours=2)))}
        )
    with pytest.raises(ValidationError, match="one path segment"):
        JobVideo.model_validate({**video, "filename": "../clip.mp4"})
    with pytest.raises(ValidationError):
        JobVideo.model_validate({**video, "mimeType": "application/octet-stream"})


def test_result_contract_requires_complete_uuid_assignment() -> None:
    with pytest.raises(ValidationError, match="requires tag, user and assignment ids"):
        JobResultManifest(
            jobId="job-1",
            status="processed",
            processedAt=datetime(2026, 8, 12, tzinfo=timezone.utc),
            algorithmVersion="test-v1",
            selectedUserId=uuid4(),
            reason="incomplete assignment",
        )
