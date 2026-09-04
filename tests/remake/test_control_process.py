import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from bearvision.control import (
    _run_until_shutdown,
    hardware,
    process_control_command,
    readiness,
    simulate,
)
from bearvision.edge import OrchestrationEvent
from bearvision.contracts import serialize_runtime_event


ROOT = Path(__file__).resolve().parents[2]


def test_control_process_replays_versioned_scenario_events(monkeypatch, capsys) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("bearvision.simulation.execution.time.sleep", sleeps.append)

    exit_code = simulate(
        ROOT / "specs" / "scenarios" / "single-rider-success.yaml",
        run_id="run-simulation-9",
        realtime=True,
        speed=2.0,
    )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert exit_code == 0
    assert events[0]["control_event_version"] == "1.1"
    assert all(event["run_id"] == "run-simulation-9" for event in events)
    assert all(event["emitted_at"].endswith("Z") for event in events)
    assert any(event["kind"] == "clip_uploaded" for event in events)
    assert all(event["kind"] != "job_published" for event in events)
    assert all(event["kind"] != "server_assignment" for event in events)
    assert sum(sleeps) == 4.5


def test_runtime_event_contract_rejects_unknown_or_malformed_events() -> None:
    with pytest.raises(ValidationError):
        serialize_runtime_event(  # type: ignore[arg-type]
            "person_detected",
            {"frame_id": "frame-1"},
            run_id="run-invalid-event",
            emitted_at=datetime(2026, 9, 3, 8, 15, tzinfo=timezone.utc),
            at_s=1,
        )
    with pytest.raises(ValidationError):
        serialize_runtime_event(  # type: ignore[arg-type]
            "invented_event",
            {},
            run_id="run-invalid-event",
            emitted_at=datetime(2026, 9, 3, 8, 15, tzinfo=timezone.utc),
            at_s=1,
        )


def test_runtime_event_contract_exposes_typed_lifecycle_and_failures() -> None:
    emitted_at = datetime(2026, 9, 3, 8, 15, tzinfo=timezone.utc)
    lifecycle = json.loads(
        serialize_runtime_event(
            "lifecycle_changed",
            {
                "stage": "uploading",
                "operation_id": "capture-frame-1:publish",
            },
            run_id="run-edge-17",
            emitted_at=emitted_at,
        )
    )
    failure = json.loads(
        serialize_runtime_event(
            "component_failed",
            {
                "failure_id": "failure-capture-frame-1-publish",
                "operation_id": "capture-frame-1:publish",
                "stage": "uploading",
                "component": "job_queue",
                "error": "Box is temporarily offline",
                "operator_message": "The clip could not be uploaded.",
                "corrective_action": "Check the network and Box connection, then retry.",
                "severity": "blocking",
                "retryable": True,
            },
            run_id="run-edge-17",
            emitted_at=emitted_at,
        )
    )

    assert lifecycle["payload"]["stage"] == "uploading"
    assert lifecycle["run_id"] == "run-edge-17"
    assert lifecycle["emitted_at"] == "2026-09-03T08:15:00Z"
    assert failure["payload"]["retryable"] is True
    with pytest.raises(ValidationError):
        serialize_runtime_event(
            "lifecycle_changed",
            {"stage": "made_up", "operation_id": None},
            run_id="run-edge-17",
            emitted_at=emitted_at,
        )


def test_hardware_uses_explicit_runtime_directories(monkeypatch) -> None:
    config_path = Path("config/production-edge.yaml")
    capture_dir = Path("state/captures")
    scratch_dir = Path("state/scratch")
    received: dict[str, Path] = {}

    class Orchestrator:
        async def run(self) -> None:
            return None

    def build_orchestrator(config, *, capture_dir, scratch_dir, event_sink):
        received["capture_dir"] = capture_dir
        received["scratch_dir"] = scratch_dir
        received["event_sink"] = event_sink
        return Orchestrator()

    monkeypatch.setattr(
        "bearvision.control.load_edge_config",
        lambda path: SimpleNamespace(system=SimpleNamespace(log_level="INFO")),
    )
    monkeypatch.setattr("bearvision.control.build_real_orchestrator", build_orchestrator)

    exit_code = asyncio.run(
        hardware(
            config_path,
            run_id="run-hardware-directories",
            capture_dir=capture_dir,
            scratch_dir=scratch_dir,
        )
    )

    assert exit_code == 0
    assert received["capture_dir"] == capture_dir
    assert received["scratch_dir"] == scratch_dir
    assert callable(received["event_sink"])


def test_control_process_exposes_python_owned_readiness(monkeypatch) -> None:
    expected = {
        "readiness_schema_version": "1.0",
        "blocking": False,
        "warning_ids": [],
        "checks": [],
    }
    monkeypatch.setattr(
        "bearvision.control.check_edge_readiness",
        lambda *args, **kwargs: SimpleNamespace(to_dict=lambda: expected),
    )

    assert readiness(
        Path("config/edge.yaml"),
        capture_dir=Path("state/captures"),
        scratch_dir=Path("state/scratch"),
    ) == expected


def test_hardware_streams_orchestrator_lifecycle_events(monkeypatch, capsys) -> None:
    class Orchestrator:
        async def run(self) -> None:
            self.event_sink(
                OrchestrationEvent(
                    at_monotonic_s=1,
                    kind="lifecycle_changed",
                    payload={"stage": "monitoring", "operation_id": None},
                )
            )

    def build_orchestrator(config, *, capture_dir, scratch_dir, event_sink):
        orchestrator = Orchestrator()
        orchestrator.event_sink = event_sink
        return orchestrator

    monkeypatch.setattr(
        "bearvision.control.load_edge_config",
        lambda path: SimpleNamespace(system=SimpleNamespace(log_level="INFO")),
    )
    monkeypatch.setattr("bearvision.control.build_real_orchestrator", build_orchestrator)

    assert asyncio.run(
        hardware(Path("config/edge.yaml"), run_id="run-hardware-4")
    ) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all(event["run_id"] == "run-hardware-4" for event in events)
    assert all(event["emitted_at"].endswith("Z") for event in events)
    assert any(
        event["kind"] == "lifecycle_changed"
        and event["payload"]["stage"] == "monitoring"
        for event in events
    )


def test_versioned_retry_command_targets_one_retained_failure() -> None:
    class Orchestrator:
        def __init__(self) -> None:
            self.failure_ids = []

        async def retry_failure(self, failure_id: str):
            self.failure_ids.append(failure_id)
            return ()

    orchestrator = Orchestrator()
    asyncio.run(
        process_control_command(
            orchestrator,
            {
                "command_version": "1.0",
                "kind": "retry_failure",
                "failure_id": "failure-upload",
            },
        )
    )

    assert orchestrator.failure_ids == ["failure-upload"]
    with pytest.raises(ValueError, match="unsupported control command version"):
        asyncio.run(
            process_control_command(
                orchestrator,
                {
                    "command_version": "9.0",
                    "kind": "retry_failure",
                    "failure_id": "failure-upload",
                },
        )
    )


def test_versioned_stop_command_runs_orchestrator_cleanup() -> None:
    async def exercise() -> None:
        shutdown_requested = asyncio.Event()

        class Orchestrator:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.cleaned_up = False

            async def run(self) -> None:
                try:
                    self.started.set()
                    await asyncio.Event().wait()
                finally:
                    self.cleaned_up = True

        orchestrator = Orchestrator()
        runtime = asyncio.create_task(
            _run_until_shutdown(orchestrator, shutdown_requested)
        )
        await orchestrator.started.wait()

        await process_control_command(
            orchestrator,
            {"command_version": "1.0", "kind": "stop_runtime"},
            shutdown_requested=shutdown_requested,
        )
        await runtime

        assert orchestrator.cleaned_up

    asyncio.run(exercise())
