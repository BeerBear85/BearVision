from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from bearvision.contracts import (
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
        battery_percent=87,
    )

    assert observation.contract_schema_version == "1.0"
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
    assert schema["properties"]["contract_schema_version"]["const"] == "1.0"
    assert "rssi_dbm" in schema["properties"]
