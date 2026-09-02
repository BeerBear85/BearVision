"""Closed-loop Edge queue and server-worker behavioural scenarios."""

from __future__ import annotations

import asyncio

from bearvision.config import AssignmentConfig
from bearvision.contracts import (
    JobResultManifest,
    PersonDetection,
    ScenarioDefinition,
    ScenarioSourceProfile,
    TagObservation,
    TagRegistryEntry,
)
from bearvision.edge.orchestrator import BearVisionOrchestrator, OrchestrationResult
from bearvision.ports import ComponentError, JobQueue, VideoFrame
from bearvision.server import ServerWorker

from .adapters import (
    InMemoryJobQueue,
    SimulatedCamera,
    SimulatedDetector,
    SimulatedTagScanner,
    VirtualClock,
)
from .scenario_runtime import (
    ScenarioRunResult,
    TraceEvent,
    build_scenario_worker,
    finalize_scenario_run,
)
from .scenario_inputs import generate_bear_tag_series


class ClosedLoopScenarioRunner:
    def __init__(
        self,
        scenario: ScenarioDefinition,
        *,
        orchestrator: BearVisionOrchestrator,
        worker: ServerWorker | None,
        clock: VirtualClock,
        observations: tuple[TagObservation, ...],
        camera: SimulatedCamera,
        queue: JobQueue,
    ) -> None:
        self.scenario = scenario
        self.orchestrator = orchestrator
        self.worker = worker
        self.clock = clock
        self.observations = observations
        self.camera = camera
        self.queue = queue

    @classmethod
    def from_scenario(
        cls,
        scenario: ScenarioDefinition,
        *,
        assignment_policy: AssignmentConfig | None = None,
        recording_duration_s: float = 5.0,
        job_queue: JobQueue | None = None,
        process_server: bool = True,
    ) -> "ClosedLoopScenarioRunner":
        if recording_duration_s <= 0:
            raise ValueError("recording_duration_s must be positive")
        if scenario.source_profile is not ScenarioSourceProfile.SYNTHETIC:
            raise ValueError("closed-loop runner requires the synthetic source profile")
        clock = VirtualClock()
        generated_observations, generated_registry = generate_bear_tag_series(
            scenario.synthetic_bear_tags,
            clock,
        )
        observations: list[TagObservation] = list(generated_observations)
        registry_by_tag = {entry.tag_id: entry for entry in generated_registry}
        detections: dict[str, tuple[PersonDetection, ...]] = {}
        for index, item in enumerate(scenario.timeline):
            observation = item.to_tag_observation(clock.start_utc)
            if observation is not None:
                observations.append(observation)
            registry_entry = item.to_registry_entry()
            if registry_entry is not None:
                registry_by_tag[registry_entry.tag_id] = registry_entry
            frame_id = f"frame-{index}"
            detection = item.to_person_detection(frame_id)
            if detection is not None:
                detections[frame_id] = (detection,)
        registry_entries: tuple[TagRegistryEntry, ...] = tuple(
            registry_by_tag[tag_id] for tag_id in sorted(registry_by_tag)
        )
        camera = SimulatedCamera(clock, fail_capture=scenario.faults.camera_capture)
        queue: JobQueue = job_queue or InMemoryJobQueue(
            fail_publish=scenario.faults.storage_upload
        )
        orchestrator = BearVisionOrchestrator(
            clock=clock,
            camera=camera,
            scanner=SimulatedTagScanner(()),
            detector=SimulatedDetector(detections),
            job_queue=queue,
            edge_device_id="scenario-edge",
            recording_duration_s=recording_duration_s,
            observation_retention_s=max(30.0, scenario.duration_s + recording_duration_s),
            ble_logging_enabled=False,
        )
        return cls(
            scenario,
            orchestrator=orchestrator,
            worker=build_scenario_worker(
                entries=registry_entries,
                queue=queue,
                clock=clock,
                assignment_policy=assignment_policy,
                enabled=process_server,
            ),
            clock=clock,
            observations=tuple(sorted(observations, key=lambda item: item.observed_at_monotonic_s)),
            camera=camera,
            queue=queue,
        )

    def run(self) -> ScenarioRunResult:
        return asyncio.run(self._run())

    async def _run(self) -> ScenarioRunResult:
        trace_events: list[TraceEvent] = []
        edge_results: dict[str, OrchestrationResult] = {}
        failures: list[dict[str, str]] = []
        detection_times: list[float] = []
        for observation in self.observations:
            self.orchestrator.add_tag_observation(observation)
        await self.orchestrator.start()
        try:
            for index, item in sorted(
                enumerate(self.scenario.timeline), key=lambda pair: (pair[1].at_s, pair[0])
            ):
                if item.is_tag_observation:
                    trace_events.append((item.at_s, item.event, item.trace_payload()))
                    continue
                if self.clock.monotonic() < item.at_s:
                    self.clock.advance_to(item.at_s)
                frame = VideoFrame(f"frame-{index}", item.at_s, 1920, 1080, b"simulated-frame")
                try:
                    evaluation = await self.orchestrator.evaluate_frame(frame)
                except ComponentError as exc:
                    component = "job_queue" if self.scenario.faults.storage_upload else "camera"
                    failures.append({"component": component, "error": str(exc)})
                    continue
                if evaluation.events:
                    detection_times.append(item.at_s)
                    trace_events.extend(
                        (event.at_monotonic_s, event.kind, event.payload)
                        for event in evaluation.events
                    )
                if evaluation.result is not None:
                    edge_results.setdefault(evaluation.result.request_id, evaluation.result)
                    failures.extend(
                        {
                            "component": str(event.payload.get("component", "runtime")),
                            "error": str(event.payload.get("error", "runtime failed")),
                        }
                        for event in evaluation.result.events
                        if event.kind == "component_failed"
                    )
        finally:
            await self.orchestrator.stop()

        server_results: list[JobResultManifest] = []
        for edge_result in edge_results.values():
            trace_events.extend(
                (event.at_monotonic_s, event.kind, event.payload)
                for event in edge_result.events
            )
            if self.worker is not None:
                server_result = await self.worker.run_once()
                if server_result is not None:
                    server_results.append(server_result)
                    trace_events.append(
                        (
                            edge_result.clip_end_monotonic_s,
                            "server_assignment",
                            server_result.model_dump(mode="json", by_alias=True),
                        )
                    )
        if self.worker is not None:
            while True:
                server_result = await self.worker.run_once()
                if server_result is None:
                    break
                server_results.append(server_result)
                trace_events.append(
                    (
                        self.clock.monotonic(),
                        "server_assignment",
                        server_result.model_dump(mode="json", by_alias=True),
                    )
                )
        captures = tuple(
            capture.media.asset.asset_id for capture in self.camera.captures.values()
        )
        return finalize_scenario_run(
            scenario=self.scenario,
            trace_events=trace_events,
            assignments=server_results,
            captures=captures,
            edge_results=edge_results.values(),
            failures=failures,
            detection_times_s=tuple(detection_times),
            evaluate_server=self.worker is not None,
        )
