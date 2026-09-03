"""Composition root for executable behavioural Scenario profiles."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from bearvision.config import AssignmentConfig, EdgeConfig
from bearvision.contracts import ScenarioDefinition, ScenarioSourceProfile
from bearvision.ports import JobQueue

if TYPE_CHECKING:
    from .runner import ClosedLoopScenarioRunner
    from .video_runner import VideoScenarioRunner


def build_behavioral_system(
    scenario: ScenarioDefinition,
    server_assignment_policy: AssignmentConfig | None = None,
    *,
    edge_config: EdgeConfig | None = None,
    capture_dir: Path | None = None,
    job_queue: JobQueue | None = None,
    process_server: bool = True,
) -> ClosedLoopScenarioRunner | VideoScenarioRunner:
    """Select the executable runner owned by the Scenario source profile."""

    if scenario.source_profile is ScenarioSourceProfile.RECORDED_VIDEO:
        from .video_runner import VideoScenarioRunner

        return VideoScenarioRunner.from_scenario(
            scenario,
            assignment_policy=server_assignment_policy,
            edge_config=edge_config,
            capture_dir=capture_dir,
            job_queue=job_queue,
            process_server=process_server,
        )
    from .runner import ClosedLoopScenarioRunner

    return ClosedLoopScenarioRunner.from_scenario(
        scenario,
        assignment_policy=server_assignment_policy,
        job_queue=job_queue,
        process_server=process_server,
        capture_dir=capture_dir,
    )
