"""Behavioural scenarios executed through the production orchestrator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timedelta

from bearvision.config import AssignmentConfig
from bearvision.contracts import (
    BoundingBox,
    PersonDetection,
    RiderAssignment,
    ScenarioDefinition,
    StorageReceipt,
    TagObservation,
    TagRegistryEntry,
    Vector3,
)
from bearvision.edge.orchestrator import BearVisionOrchestrator, OrchestrationResult
from bearvision.ports import ComponentError, VideoFrame

from .adapters import (
    InMemoryStorage,
    InMemoryTagRegistry,
    SimulatedCamera,
    SimulatedDetector,
    SimulatedTagScanner,
    VirtualClock,
)
from .engine import TraceEntry


@dataclass(frozen=True, slots=True)
class ScenarioRunResult:
    trace: tuple[TraceEntry, ...]
    assignments: tuple[RiderAssignment, ...]
    captures: tuple[str, ...]
    uploads: tuple[StorageReceipt, ...]
    failures: tuple[dict[str, str], ...]
    expectation_failures: tuple[str, ...] = ()


class ClosedLoopScenarioRunner:
    """Drive the exact same orchestration core used by the edge service."""

    def __init__(
        self,
        scenario: ScenarioDefinition,
        *,
        orchestrator: BearVisionOrchestrator,
        clock: VirtualClock,
        observations: tuple[TagObservation, ...],
        camera: SimulatedCamera,
        storage: InMemoryStorage,
    ) -> None:
        self.scenario = scenario
        self.orchestrator = orchestrator
        self.clock = clock
        self.observations = observations
        self.camera = camera
        self.storage = storage

    @classmethod
    def from_scenario(
        cls,
        scenario: ScenarioDefinition,
        *,
        assignment_policy: AssignmentConfig | None = None,
        recording_duration_s: float = 5.0,
    ) -> "ClosedLoopScenarioRunner":
        if recording_duration_s <= 0:
            raise ValueError("recording_duration_s must be positive")
        clock = VirtualClock()
        observations: list[TagObservation] = []
        entries: dict[str, TagRegistryEntry] = {}
        detections: dict[str, tuple[PersonDetection, ...]] = {}
        for index, item in enumerate(scenario.timeline):
            if item.event in {"tag_enters_range", "tag_observation"}:
                tag_id = str(item.payload["tag_id"])
                rider_id = item.payload.get("rider_id")
                if rider_id is not None:
                    entries[tag_id] = TagRegistryEntry(tag_id=tag_id, rider_id=str(rider_id))
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
                            x_px=100,
                            y_px=100,
                            width_px=400,
                            height_px=700,
                        ),
                        confidence=float(item.payload.get("confidence", 0.9)),
                    ),
                )

        camera = SimulatedCamera(clock, fail_capture=scenario.faults.camera_capture)
        storage = InMemoryStorage(clock, fail_upload=scenario.faults.storage_upload)
        orchestrator = BearVisionOrchestrator(
            clock=clock,
            camera=camera,
            scanner=SimulatedTagScanner(()),
            detector=SimulatedDetector(detections),
            storage=storage,
            registry=InMemoryTagRegistry(entries.values()),
            assignment_policy=assignment_policy or AssignmentConfig(),
            recording_duration_s=recording_duration_s,
            observation_retention_s=max(30.0, scenario.duration_s + recording_duration_s),
            ble_logging_enabled=False,
        )
        return cls(
            scenario,
            orchestrator=orchestrator,
            clock=clock,
            observations=tuple(sorted(observations, key=lambda item: item.observed_at_monotonic_s)),
            camera=camera,
            storage=storage,
        )

    def run(self) -> ScenarioRunResult:
        return asyncio.run(self._run())

    async def _run(self) -> ScenarioRunResult:
        trace_events: list[tuple[float, str, dict]] = []
        results: dict[str, OrchestrationResult] = {}
        failures: list[dict[str, str]] = []

        for observation in self.observations:
            self.orchestrator.add_tag_observation(observation)
        await self.orchestrator.start()
        try:
            for index, item in sorted(
                enumerate(self.scenario.timeline), key=lambda pair: (pair[1].at_s, pair[0])
            ):
                trace_events.append((item.at_s, item.event, dict(item.payload)))
                if item.event in {"tag_enters_range", "tag_observation"}:
                    trace_events.append(
                        (
                            item.at_s,
                            "tag_observed",
                            {
                                "tag_id": str(item.payload["tag_id"]),
                                "rssi_dbm": int(item.payload.get("rssi_dbm", -60)),
                                "acceleration_mps2": item.payload.get("acceleration_mps2", {}),
                            },
                        )
                    )
                    continue

                frame_id = f"frame-{index}"
                if self.clock.monotonic() < item.at_s:
                    self.clock.advance_to(item.at_s)
                frame = VideoFrame(frame_id, item.at_s, 1920, 1080, b"simulated-frame")
                try:
                    result = await self.orchestrator.process_frame(frame)
                except ComponentError as exc:
                    component = "storage" if self.scenario.faults.storage_upload else "camera"
                    failure = {"component": component, "error": str(exc)}
                    failures.append(failure)
                    trace_events.append((self.clock.monotonic(), "component_failed", failure))
                    continue
                if result is not None:
                    results.setdefault(result.request_id, result)
        finally:
            await self.orchestrator.stop()

        for result in results.values():
            trace_events.extend(
                [
                    (
                        result.clip_start_monotonic_s,
                        "capture_started",
                        {
                            "asset_id": result.media.asset.asset_id,
                            "clip_end_s": result.clip_end_monotonic_s,
                        },
                    ),
                    (
                        result.clip_end_monotonic_s,
                        "finalize_clip",
                        {"request_id": result.request_id},
                    ),
                    (
                        result.clip_end_monotonic_s,
                        "rider_assignment",
                        result.assignment.model_dump(mode="json"),
                    ),
                    (
                        result.clip_end_monotonic_s,
                        "capture_completed",
                        {"asset_id": result.media.asset.asset_id},
                    ),
                ]
            )
            if result.upload is not None:
                trace_events.append(
                    (
                        result.clip_end_monotonic_s,
                        "clip_uploaded",
                        {
                            "asset_id": result.upload.asset_id,
                            "object_key": result.upload.object_key,
                        },
                    )
                )

        ordered = sorted(enumerate(trace_events), key=lambda item: (item[1][0], item[0]))
        trace = tuple(
            TraceEntry(at_s=at_s, sequence=sequence, kind=kind, payload=payload)
            for sequence, (_, (at_s, kind, payload)) in enumerate(ordered)
        )
        assignments = tuple(result.assignment for result in results.values())
        # SimulatedCamera stores by request id; expose stable asset ids.
        captures = tuple(media.asset.asset_id for media in self.camera.captures.values())
        uploads = tuple(receipt for _, receipt in self.storage.objects.values())
        expectation_failures = self._evaluate_expectations(assignments, captures, uploads)
        return ScenarioRunResult(
            trace=trace,
            assignments=assignments,
            captures=captures,
            uploads=uploads,
            failures=tuple(failures),
            expectation_failures=expectation_failures,
        )

    def _evaluate_expectations(
        self,
        assignments: tuple[RiderAssignment, ...],
        captures: tuple[str, ...],
        uploads: tuple[StorageReceipt, ...],
    ) -> tuple[str, ...]:
        expected = self.scenario.expect
        failures: list[str] = []
        first = assignments[0] if assignments else None
        if expected.rider_id is not None and (first is None or first.rider_id != expected.rider_id):
            actual = first.rider_id if first is not None else None
            failures.append(f"expected rider_id={expected.rider_id!r}, got {actual!r}")
        if expected.assignment_status is not None and (
            first is None or first.status.value != expected.assignment_status
        ):
            actual = first.status.value if first is not None else None
            failures.append(
                f"expected assignment_status={expected.assignment_status!r}, got {actual!r}"
            )
        if expected.capture_triggered is not None and bool(captures) != expected.capture_triggered:
            failures.append(
                f"expected capture_triggered={expected.capture_triggered}, got {bool(captures)}"
            )
        if expected.clip_uploaded is not None and bool(uploads) != expected.clip_uploaded:
            failures.append(f"expected clip_uploaded={expected.clip_uploaded}, got {bool(uploads)}")
        return tuple(failures)
