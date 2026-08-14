import json
from pathlib import Path

from bearvision.control import simulate


ROOT = Path(__file__).resolve().parents[2]


def test_control_process_replays_versioned_scenario_events(monkeypatch, capsys) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("bearvision.control.time.sleep", sleeps.append)

    exit_code = simulate(
        ROOT / "specs" / "scenarios" / "single-rider-success.yaml",
        realtime=True,
        speed=2.0,
    )

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert exit_code == 0
    assert events[0]["control_event_version"] == "1.0"
    assert any(event["kind"] == "server_assignment" for event in events)
    assert sum(sleeps) == 4.5
