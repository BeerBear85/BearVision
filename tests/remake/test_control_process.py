import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from bearvision.control import hardware, simulate
from bearvision.contracts import serialize_runtime_event


ROOT = Path(__file__).resolve().parents[2]


def test_control_process_replays_versioned_scenario_events(monkeypatch, capsys) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("bearvision.simulation.execution.time.sleep", sleeps.append)

    exit_code = simulate(
        ROOT / "specs" / "scenarios" / "single-rider-success.yaml",
        realtime=True,
        speed=2.0,
    )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert exit_code == 0
    assert events[0]["control_event_version"] == "1.0"
    assert any(event["kind"] == "clip_uploaded" for event in events)
    assert all(event["kind"] != "job_published" for event in events)
    assert all(event["kind"] != "server_assignment" for event in events)
    assert sum(sleeps) == 4.5


def test_runtime_event_contract_rejects_unknown_or_malformed_events() -> None:
    with pytest.raises(ValidationError):
        serialize_runtime_event(  # type: ignore[arg-type]
            "person_detected",
            {"frame_id": "frame-1"},
            at_s=1,
        )
    with pytest.raises(ValidationError):
        serialize_runtime_event(  # type: ignore[arg-type]
            "invented_event",
            {},
            at_s=1,
        )


def test_hardware_uses_explicit_runtime_directories(monkeypatch) -> None:
    config_path = Path("config/production-edge.yaml")
    capture_dir = Path("state/captures")
    scratch_dir = Path("state/scratch")
    received: dict[str, Path] = {}

    class Orchestrator:
        async def run(self) -> None:
            return None

    def build_orchestrator(config, *, capture_dir, scratch_dir):
        received["capture_dir"] = capture_dir
        received["scratch_dir"] = scratch_dir
        return Orchestrator()

    monkeypatch.setattr(
        "bearvision.control.load_edge_config",
        lambda path: SimpleNamespace(system=SimpleNamespace(log_level="INFO")),
    )
    monkeypatch.setattr("bearvision.control.build_real_orchestrator", build_orchestrator)

    exit_code = asyncio.run(
        hardware(
            config_path,
            capture_dir=capture_dir,
            scratch_dir=scratch_dir,
        )
    )

    assert exit_code == 0
    assert received == {
        "capture_dir": capture_dir,
        "scratch_dir": scratch_dir,
    }
