from pathlib import Path

from bearvision.contracts import ScenarioDefinition, load_scenario
from bearvision.edge import build_behavioral_system
from bearvision.simulation import ClosedLoopScenarioRunner


ROOT = Path(__file__).resolve().parents[2]


def scenario(timeline, *, faults=None) -> ScenarioDefinition:
    return ScenarioDefinition(
        scenario_schema_version="2.0",
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
    assert first.assignments[0].selected_user_email == "rider-17@scenario.invalid"
    assert first.assignments[0].candidates[0].observation_count == 2
    assert len(first.captures) == 1
    assert first.uploads[0].object_key.startswith("input-queue/ready/")
    assert not first.failures


def test_no_tag_is_unassigned_but_detection_is_still_captured() -> None:
    result = ClosedLoopScenarioRunner.from_scenario(
        scenario([{"at_s": 2, "event": "person_detected", "payload": {"confidence": 0.9}}])
    ).run()
    assert result.assignments[0].status == "unresolved"
    assert result.assignments[0].error_code == "NO_QUALIFIED_BEARTAG"


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
                {
                    "at_s": 2.1,
                    "event": "tag_observation",
                    "payload": {"tag_id": "tag-17", "rssi_dbm": -50, "acceleration_mps2": {"x": 0, "y": 0, "z": 19}},
                },
                {
                    "at_s": 2.2,
                    "event": "tag_observation",
                    "payload": {"tag_id": "tag-17", "rssi_dbm": -51, "acceleration_mps2": {"x": 0, "y": 0, "z": 19}},
                },
                {
                    "at_s": 2.1,
                    "event": "tag_observation",
                    "payload": {"tag_id": "tag-22", "rssi_dbm": -55, "acceleration_mps2": {"x": 0, "y": 0, "z": 19}},
                },
                {
                    "at_s": 2.2,
                    "event": "tag_observation",
                    "payload": {"tag_id": "tag-22", "rssi_dbm": -56, "acceleration_mps2": {"x": 0, "y": 0, "z": 19}},
                },
            ]
        )
    ).run()
    assert result.assignments[0].status == "unresolved"
    assert result.assignments[0].error_code == "AMBIGUOUS_BEARTAG"
    assert result.assignments[0].selected_user_email is None


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
                {
                    "at_s": 3.1,
                    "event": "tag_observation",
                    "payload": {"tag_id": "active", "rssi_dbm": -65, "acceleration_mps2": {"x": 4, "y": 2, "z": 19}},
                },
                {
                    "at_s": 3.2,
                    "event": "tag_observation",
                    "payload": {"tag_id": "active", "rssi_dbm": -66, "acceleration_mps2": {"x": 4, "y": 2, "z": 18}},
                },
                {
                    "at_s": 3.1,
                    "event": "tag_observation",
                    "payload": {"tag_id": "nearby", "rssi_dbm": -40, "acceleration_mps2": {"x": 0, "y": 0, "z": 9.81}},
                },
                {
                    "at_s": 3.2,
                    "event": "tag_observation",
                    "payload": {"tag_id": "nearby", "rssi_dbm": -41, "acceleration_mps2": {"x": 0, "y": 0, "z": 9.8}},
                },
            ]
        )
    ).run()
    assert result.assignments[0].selected_user_email == "rider-active@scenario.invalid"
    assert result.uploads[0].object_key.startswith("input-queue/ready/")


def test_assignment_uses_accelerometer_samples_from_entire_clip() -> None:
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
                {
                    "at_s": 7.5,
                    "event": "tag_observation",
                    "payload": {
                        "tag_id": "tag-17",
                        "rssi_dbm": -53,
                        "acceleration_mps2": {"x": 3, "y": 2, "z": 18},
                    },
                },
            ]
        )
    ).run()
    assert result.assignments[0].selected_user_email == "rider-17@scenario.invalid"
    assert result.assignments[0].candidates[0].observation_count == 2
    assert any(item.kind == "server_assignment" for item in result.trace)


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
    assert result.failures[0]["component"] == "job_queue"


def test_declared_scenario_expectations_are_executable() -> None:
    definition = ScenarioDefinition(
        scenario_schema_version="2.0",
        name="wrong-rider-expectation",
        duration_s=20,
        timeline=[
            {
                "at_s": 2,
                "event": "person_detected",
                "payload": {"confidence": 0.9},
            }
        ],
        expect={"rider_id": "rider-that-was-not-observed"},
    )
    result = ClosedLoopScenarioRunner.from_scenario(definition).run()
    assert result.expectation_failures == (
        "expected rider_id='rider-that-was-not-observed', got None",
    )
