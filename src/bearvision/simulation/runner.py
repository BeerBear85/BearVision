"""Closed-loop Edge queue and server-worker behavioural scenarios."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timedelta
from uuid import NAMESPACE_URL, UUID, uuid5

from bearvision.config import AssignmentConfig
from bearvision.contracts import (
    BoundingBox,
    JobResultManifest,
    PersonDetection,
    ScenarioDefinition,
    StorageReceipt,
    TagObservation,
    Vector3,
)
from bearvision.edge.orchestrator import BearVisionOrchestrator, OrchestrationResult
from bearvision.ports import ComponentError, JobQueue, VideoFrame
from bearvision.server import (
    BearTagAssignment,
    BearTagRecord,
    InMemoryUserRegistry,
    RegistryData,
    ServerWorker,
    UserRecord,
    normalize_user_email,
)

from .adapters import (
    InMemoryJobQueue,
    SimulatedCamera,
    SimulatedDetector,
    SimulatedTagScanner,
    VirtualClock,
)
from .engine import TraceEntry


def _scenario_email(rider_id: str) -> str:
    return normalize_user_email(rider_id if "@" in rider_id else f"{rider_id}@scenario.invalid")


def _scenario_user_id(rider_id: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"bearvision:scenario-user:{_scenario_email(rider_id)}")


@dataclass(frozen=True, slots=True)
class ScenarioRunResult:
    trace: tuple[TraceEntry, ...]
    assignments: tuple[JobResultManifest, ...]
    captures: tuple[str, ...]
    uploads: tuple[StorageReceipt, ...]
    failures: tuple[dict[str, str], ...]
    expectation_failures: tuple[str, ...] = ()


def evaluate_expectations(
    scenario: ScenarioDefinition,
    assignments: tuple[JobResultManifest, ...],
    captures: tuple[str, ...],
    uploads: tuple[StorageReceipt, ...],
    *,
    detection_times_s: tuple[float, ...] = (),
    evaluate_server: bool = True,
) -> tuple[str, ...]:
    expected = scenario.expect
    failures: list[str] = []
    first = assignments[0] if assignments else None
    if evaluate_server and expected.rider_id is not None:
        expected_user_id = _scenario_user_id(expected.rider_id)
        if first is None or first.selected_user_id != expected_user_id:
            actual = first.selected_user_id if first is not None else None
            failures.append(f"expected rider_id={expected.rider_id!r}, got {actual!r}")
    if evaluate_server and expected.assignment_status is not None:
        actual_status: str | None = None
        if first is not None:
            if first.status == "processed":
                actual_status = "assigned"
            elif first.error_code == "AMBIGUOUS_BEARTAG":
                actual_status = "ambiguous"
            else:
                actual_status = "unassigned"
        if actual_status != expected.assignment_status:
            failures.append(
                f"expected assignment_status={expected.assignment_status!r}, "
                f"got {actual_status!r}"
            )
    if expected.capture_triggered is not None and bool(captures) != expected.capture_triggered:
        failures.append(
            f"expected capture_triggered={expected.capture_triggered}, got {bool(captures)}"
        )
    if expected.clip_uploaded is not None and bool(uploads) != expected.clip_uploaded:
        failures.append(f"expected clip_uploaded={expected.clip_uploaded}, got {bool(uploads)}")
    if (
        expected.minimum_person_detections is not None
        and len(detection_times_s) < expected.minimum_person_detections
    ):
        failures.append(
            f"expected at least {expected.minimum_person_detections} person detections, "
            f"got {len(detection_times_s)}"
        )
    if expected.first_detection_between_s is not None:
        start, end = expected.first_detection_between_s
        detection_at = detection_times_s[0] if detection_times_s else None
        if detection_at is None or not start <= detection_at <= end:
            failures.append(f"expected first detection in [{start}, {end}], got {detection_at!r}")
    return tuple(failures)


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
        users = tuple(
            UserRecord(
                id=_scenario_user_id(rider),
                email=_scenario_email(rider),
                displayName=rider,
            )
            for rider in sorted(set(riders_by_tag.values()))
        )
        tags = tuple(BearTagRecord(id=tag_id) for tag_id in sorted(riders_by_tag))
        assignments = tuple(
            BearTagAssignment(
                id=f"scenario-{tag_id}",
                userId=_scenario_user_id(rider),
                bearTagId=tag_id,
                validFrom=clock.start_utc - timedelta(days=1),
                validTo=clock.start_utc + timedelta(days=1),
            )
            for tag_id, rider in sorted(riders_by_tag.items())
        )
        registry = InMemoryUserRegistry(
            RegistryData(users=users, bearTags=tags, assignments=assignments)
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
            worker=(
                ServerWorker(queue, registry, clock, assignment_policy)
                if process_server
                else None
            ),
            clock=clock,
            observations=tuple(sorted(observations, key=lambda item: item.observed_at_monotonic_s)),
            camera=camera,
            queue=queue,
        )

    def run(self) -> ScenarioRunResult:
        return asyncio.run(self._run())

    async def _run(self) -> ScenarioRunResult:
        trace_events: list[tuple[float, str, dict]] = []
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
        ordered = sorted(enumerate(trace_events), key=lambda item: (item[1][0], item[0]))
        trace = tuple(
            TraceEntry(at_s=at_s, sequence=sequence, kind=kind, payload=payload)
            for sequence, (_, (at_s, kind, payload)) in enumerate(ordered)
        )
        captures = tuple(media.asset.asset_id for media in self.camera.captures.values())
        uploads = tuple(
            StorageReceipt(
                asset_id=result.media.asset.asset_id,
                object_key=f"input-queue/ready/{result.request_id}",
                stored_at_utc=result.manifest.created_at,
                checksum_sha256=result.manifest.video.sha256,
            )
            for result in edge_results.values()
            if result.published
        )
        expectation_failures = evaluate_expectations(
            self.scenario,
            tuple(server_results),
            captures,
            uploads,
            detection_times_s=tuple(detection_times),
            evaluate_server=self.worker is not None,
        )
        return ScenarioRunResult(
            trace=trace,
            assignments=tuple(server_results),
            captures=captures,
            uploads=uploads,
            failures=tuple(failures),
            expectation_failures=expectation_failures,
        )
