"""Closed-loop Edge queue and server-worker behavioural scenarios."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from bearvision.config import AssignmentConfig
from bearvision.contracts import (
    BoundingBox,
    JobResultManifest,
    PersonDetection,
    ScenarioDefinition,
    TagObservation,
    TagRegistryEntry,
    Vector3,
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
        clock = VirtualClock()
        observations: list[TagObservation] = []
        riders_by_tag: dict[str, str] = {}
        detections: dict[str, tuple[PersonDetection, ...]] = {}
        for index, item in enumerate(scenario.timeline):
            if item.event in {"tag_enters_range", "tag_observation"}:
                tag_id = str(item.payload["tag_id"])
                if item.payload.get("rider_id") is not None:
                    riders_by_tag[tag_id] = str(item.payload["rider_id"])
                acceleration = item.payload.get("acceleration_mps2", {})
                observations.append(
                    TagObservation(
                        tag_id=tag_id,
                        observed_at_utc=clock.start_utc + timedelta(seconds=item.at_s),
                        observed_at_monotonic_s=item.at_s,
                        rssi_dbm=int(item.payload.get("rssi_dbm", -60)),
                        acceleration_mps2=Vector3(
                            x=float(acceleration.get("x", 0)),
                            y=float(acceleration.get("y", 0)),
                            z=float(acceleration.get("z", 9.80665)),
                        ),
                        battery_voltage_mv=item.payload.get("battery_voltage_mv"),
                    )
                )
            else:
                frame_id = f"frame-{index}"
                detections[frame_id] = (
                    PersonDetection(
                        frame_id=frame_id,
                        observed_at_monotonic_s=item.at_s,
                        bounding_box=BoundingBox(
                            x_px=100, y_px=100, width_px=400, height_px=700
                        ),
                        confidence=float(item.payload.get("confidence", 0.9)),
                    ),
                )
        registry_entries = tuple(
            TagRegistryEntry(tag_id=tag_id, rider_id=rider_id)
            for tag_id, rider_id in sorted(riders_by_tag.items())
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
                trace_events.append((item.at_s, item.event, dict(item.payload)))
                if item.event in {"tag_enters_range", "tag_observation"}:
                    continue
                if self.clock.monotonic() < item.at_s:
                    self.clock.advance_to(item.at_s)
                frame = VideoFrame(f"frame-{index}", item.at_s, 1920, 1080, b"simulated-frame")
                detection_times.append(item.at_s)
                try:
                    result = await self.orchestrator.process_frame(frame)
                except ComponentError as exc:
                    component = "job_queue" if self.scenario.faults.storage_upload else "camera"
                    failures.append({"component": component, "error": str(exc)})
                    continue
                if result is not None:
                    edge_results.setdefault(result.request_id, result)
        finally:
            await self.orchestrator.stop()

        server_results: list[JobResultManifest] = []
        if self.worker is not None:
            while True:
                processed_result = await self.worker.run_once()
                if processed_result is None:
                    break
                server_results.append(processed_result)
        for edge_result in edge_results.values():
            trace_events.extend(
                [
                    (
                        edge_result.clip_start_monotonic_s,
                        "capture_started",
                        {"job_id": edge_result.request_id},
                    ),
                    (
                        edge_result.clip_end_monotonic_s,
                        "job_published",
                        edge_result.manifest.model_dump(mode="json", by_alias=True),
                    ),
                    (
                        edge_result.clip_end_monotonic_s,
                        "capture_completed",
                        {"asset_id": edge_result.media.asset.asset_id},
                    ),
                ]
            )
        for server_result in server_results:
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
