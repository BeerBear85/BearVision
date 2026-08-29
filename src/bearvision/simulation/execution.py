"""One execution and replay interface for every Scenario presentation adapter."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from bearvision.config import load_edge_config
from bearvision.contracts import RuntimeEventKind, load_scenario
from bearvision.edge import build_behavioral_system
from bearvision.server import FileSystemJobQueue

from .scenario_runtime import ScenarioRunResult


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    at_s: float | None
    kind: RuntimeEventKind
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReplayOptions:
    realtime: bool = False
    speed: float = 1.0
    include_server_assignments: bool = True

    def __post_init__(self) -> None:
        if self.speed <= 0:
            raise ValueError("speed must be positive")


@dataclass(frozen=True, slots=True)
class ScenarioExecution:
    """Completed Scenario with deterministic presentation-neutral replay."""

    result: ScenarioRunResult

    @classmethod
    def run(
        cls,
        scenario_path: Path,
        *,
        config_path: Path = Path("config/edge.yaml"),
        local_queue_root: Path | None = None,
    ) -> "ScenarioExecution":
        queue = FileSystemJobQueue(local_queue_root) if local_queue_root else None
        result = build_behavioral_system(
            load_scenario(scenario_path),
            edge_config=load_edge_config(config_path),
            job_queue=queue,
            process_server=queue is None,
        ).run()
        return cls(result)

    @property
    def exit_code(self) -> int:
        return 1 if self.result.failures or self.result.expectation_failures else 0

    def replay(self, options: ReplayOptions) -> Iterator[ReplayEvent]:
        previous_at_s = 0.0
        for entry in self.result.trace:
            if not options.include_server_assignments and entry.kind == "server_assignment":
                continue
            if options.realtime:
                time.sleep(max(0.0, entry.at_s - previous_at_s) / options.speed)
            previous_at_s = entry.at_s
            yield ReplayEvent(entry.at_s, entry.kind, dict(entry.payload))
        for message in self.result.expectation_failures:
            yield ReplayEvent(None, "expectation_failed", {"message": message})
        for failure in self.result.failures:
            yield ReplayEvent(None, "component_failed", dict(failure))
