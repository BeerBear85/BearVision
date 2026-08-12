from pathlib import Path

from bearvision.contracts import RiderAssignmentStatus, ScenarioDefinition, load_scenario
from bearvision.edge import build_behavioral_system
from bearvision.simulation import ClosedLoopScenarioRunner


ROOT = Path(__file__).resolve().parents[2]


def scenario(timeline, *, faults=None) -> ScenarioDefinition:
    return ScenarioDefinition(
        scenario_schema_version="1.0",
        name="test-scenario",
        seed=42,
        duration_s=20,
        timeline=timeline,
        faults=faults or {},
        expect={},
    )


def test_single_rider_scenario_runs_closed_loop_deterministically() -> None:
    definition = load_scenario(ROOT / "specs" / "scenarios" / "single-rider-success.yaml")
    first = build_behavioral_system(definition).run()
    second = build_behavioral_system(definition).run()

    assert first.trace == second.trace
    assert first.assignments[0].rider_id == "rider-17"
    assert len(first.captures) == 1
    assert first.uploads[0].object_key.startswith("rider-17/")
    assert not first.failures


def test_no_tag_is_unassigned_but_detection_is_still_captured() -> None:
    result = ClosedLoopScenarioRunner.from_scenario(
        scenario([{"at_s": 2, "event": "person_detected", "payload": {"confidence": 0.9}}])
    ).run()
    assert result.assignments[0].status is RiderAssignmentStatus.UNASSIGNED
    assert result.uploads[0].object_key.startswith("unassigned/")


def test_two_tags_remain_ambiguous() -> None:
    result = ClosedLoopScenarioRunner.from_scenario(
        scenario(
            [
                {
                    "at_s": 1,
                    "event": "tag_enters_range",
                    "payload": {
                        "tag_id": "tag-17",
                        "rider_id": "rider-17",
                        "rssi_dbm": -50,
                        "acceleration_mps2": {"x": 0, "y": 0, "z": 19},
                    },
                },
                {
                    "at_s": 1.5,
                    "event": "tag_enters_range",
                    "payload": {
                        "tag_id": "tag-22",
                        "rider_id": "rider-22",
                        "rssi_dbm": -55,
                        "acceleration_mps2": {"x": 0, "y": 0, "z": 19},
                    },
                },
                {"at_s": 2, "event": "person_detected", "payload": {"confidence": 0.9}},
            ]
        )
    ).run()
    assert result.assignments[0].status is RiderAssignmentStatus.AMBIGUOUS
    assert result.assignments[0].rider_id is None
    assert result.uploads[0].object_key.startswith("ambiguous/")


def test_active_rider_beats_stronger_stationary_nearby_tag_in_closed_loop() -> None:
    result = ClosedLoopScenarioRunner.from_scenario(
        scenario(
            [
                {
                    "at_s": 2.5,
                    "event": "tag_observation",
                    "payload": {
                        "tag_id": "active",
                        "rider_id": "rider-active",
                        "rssi_dbm": -65,
                        "acceleration_mps2": {"x": 4, "y": 2, "z": 19},
                    },
                },
                {
                    "at_s": 2.6,
                    "event": "tag_observation",
                    "payload": {
                        "tag_id": "nearby",
                        "rider_id": "rider-nearby",
                        "rssi_dbm": -40,
                        "acceleration_mps2": {"x": 0, "y": 0, "z": 9.81},
                    },
                },
                {"at_s": 3, "event": "person_detected", "payload": {"confidence": 0.9}},
            ]
        )
    ).run()
    assert result.assignments[0].rider_id == "rider-active"
    assert result.uploads[0].object_key.startswith("rider-active/")


def test_assignment_waits_for_accelerometer_samples_after_jump_timestamp() -> None:
    result = ClosedLoopScenarioRunner.from_scenario(
        scenario(
            [
                {
                    "at_s": 1,
                    "event": "tag_enters_range",
                    "payload": {
                        "tag_id": "tag-17",
                        "rider_id": "rider-17",
                        "rssi_dbm": -50,
                        "acceleration_mps2": {"x": 0, "y": 0, "z": 9.81},
                    },
                },
                {"at_s": 3, "event": "person_detected", "payload": {"confidence": 0.9}},
                {
                    "at_s": 3.5,
                    "event": "tag_observation",
                    "payload": {
                        "tag_id": "tag-17",
                        "rssi_dbm": -52,
                        "acceleration_mps2": {"x": 4, "y": 2, "z": 19},
                    },
                },
            ]
        )
    ).run()
    assert result.assignments[0].rider_id == "rider-17"
    assert result.assignments[0].assigned_at_monotonic_s == 3.75
    assert any(item.kind == "evaluate_rider_assignment" for item in result.trace)


def test_camera_failure_stops_before_upload() -> None:
    result = ClosedLoopScenarioRunner.from_scenario(
        scenario(
            [{"at_s": 2, "event": "person_detected", "payload": {"confidence": 0.9}}],
            faults={"camera_capture": True},
        )
    ).run()
    assert not result.captures
    assert not result.uploads
    assert result.failures[0]["component"] == "camera"


def test_storage_failure_preserves_completed_capture() -> None:
    result = ClosedLoopScenarioRunner.from_scenario(
        scenario(
            [{"at_s": 2, "event": "person_detected", "payload": {"confidence": 0.9}}],
            faults={"storage_upload": True},
        )
    ).run()
    assert len(result.captures) == 1
    assert not result.uploads
    assert result.failures[0]["component"] == "storage"
