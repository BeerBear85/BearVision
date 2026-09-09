from pathlib import Path

from bearvision.simulation import ReplayOptions, ScenarioExecution


ROOT = Path(__file__).resolve().parents[2]


def test_scenario_execution_owns_replay_filtering_timing_and_status(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("bearvision.simulation.execution.time.sleep", sleeps.append)
    execution = ScenarioExecution.run(
        ROOT / "specs/scenarios/single-rider-success.yaml",
        config_path=ROOT / "config/edge.yaml",
    )

    public_events = tuple(
        execution.replay(ReplayOptions(
            realtime=True,
            speed=2,
            include_server_assignments=False,
        ))
    )
    diagnostic_events = tuple(
        execution.replay(ReplayOptions(
            realtime=False,
            speed=1,
            include_server_assignments=True,
        ))
    )

    assert execution.exit_code == 0
    assert any(event.kind == "clip_uploaded" for event in public_events)
    assert all(event.kind != "server_assignment" for event in public_events)
    assert any(event.kind == "server_assignment" for event in diagnostic_events)
    assert sum(sleeps) == 4.5


def test_scenario_execution_writes_outputs_to_an_explicit_capture_directory(tmp_path) -> None:
    capture_dir = tmp_path / "captures"

    execution = ScenarioExecution.run(
        ROOT / "specs/scenarios/single-rider-success.yaml",
        config_path=ROOT / "config/edge.yaml",
        capture_dir=capture_dir,
    )

    capture_filenames = tuple(
        entry.payload["filename"]
        for entry in execution.result.trace
        if entry.kind == "capture_completed"
    )
    assert capture_filenames
    assert all((capture_dir / filename).is_file() for filename in capture_filenames)
