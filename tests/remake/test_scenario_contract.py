from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from bearvision.contracts import ScenarioDefinition, ScenarioSourceProfile, load_scenario


def test_example_scenario_is_versioned_and_valid() -> None:
    root = Path(__file__).resolve().parents[2]
    scenario = load_scenario(root / "specs" / "scenarios" / "single-rider-success.yaml")

    assert scenario.scenario_schema_version == "2.0"
    assert scenario.timeline[0].event == "tag_enters_range"
    assert scenario.source_profile is ScenarioSourceProfile.SYNTHETIC

    observation = scenario.timeline[0].to_tag_observation(
        datetime(2026, 1, 1, tzinfo=timezone.utc)
    )

    assert observation is not None
    assert observation.tag_id == "tag-17"
    assert observation.rssi_dbm == -52


def test_timeline_payload_shape_is_validated_before_execution() -> None:
    with pytest.raises(ValidationError, match="tag_id"):
        ScenarioDefinition(
            scenario_schema_version="2.0",
            name="invalid-tag-input",
            duration_s=10,
            timeline=[
                {
                    "at_s": 1,
                    "event": "tag_observation",
                    "payload": {"confidence": 0.9},
                }
            ],
        )


def test_timeline_defaults_are_normalized_at_load() -> None:
    scenario = ScenarioDefinition(
        scenario_schema_version="2.0",
        name="normalized-input",
        duration_s=10,
        timeline=[{"at_s": 1, "event": "person_detected"}],
    )

    detection = scenario.timeline[0].to_person_detection("frame-0")

    assert detection is not None
    assert detection.confidence == 0.9
    assert scenario.timeline[0].trace_payload() == {"confidence": 0.9}
