"""Closed-loop behavioural scenario execution through component ports."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from bearvision.contracts import (
    BoundingBox,
    CaptureRequest,
    PersonDetection,
    RiderAssignment,
    ScenarioDefinition,
    StorageReceipt,
    TagObservation,
    TagRegistryEntry,
    Vector3,
)
from bearvision.config import AssignmentConfig
from bearvision.domain import assign_rider
from bearvision.ports import Camera, ComponentError, Detector, Storage, TagRegistry, TagScanner, VideoFrame

from .adapters import (
    InMemoryStorage,
    InMemoryTagRegistry,
    SimulatedCamera,
    SimulatedDetector,
    SimulatedTagScanner,
    VirtualClock,
)
from .engine import BehavioralSimulation, Event, TraceEntry


@dataclass(frozen=True, slots=True)
class ScenarioRunResult:
    trace: tuple[TraceEntry, ...]
    assignments: tuple[RiderAssignment, ...]
    captures: tuple[str, ...]
    uploads: tuple[StorageReceipt, ...]
    failures: tuple[dict[str, str], ...]


class ClosedLoopScenarioRunner:
    """Run versioned inputs through the same ports used by the edge composition."""

    def __init__(
        self,
        scenario: ScenarioDefinition,
        *,
        clock: VirtualClock,
        camera: Camera,
        scanner: TagScanner,
        detector: Detector,
        storage: Storage,
        registry: TagRegistry,
        assignment_policy: AssignmentConfig | None = None,
    ) -> None:
        self.scenario = scenario
        self.clock = clock
        self.camera = camera
        self.scanner = scanner
        self.detector = detector
        self.storage = storage
        self.registry = registry
        self.assignment_policy = assignment_policy or AssignmentConfig()

    @classmethod
    def from_scenario(
        cls,
        scenario: ScenarioDefinition,
        *,
        assignment_policy: AssignmentConfig | None = None,
    ) -> "ClosedLoopScenarioRunner":
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
                        battery_percent=item.payload.get("battery_percent"),
                    )
                )
            elif item.event == "person_detected":
                frame_id = f"frame-{index}"
                detections[frame_id] = (
                    PersonDetection(
                        frame_id=frame_id,
                        observed_at_monotonic_s=item.at_s,
                        bounding_box=BoundingBox(x_px=100, y_px=100, width_px=400, height_px=700),
                        confidence=float(item.payload.get("confidence", 0.9)),
                    ),
                )
        return cls(
            scenario,
            clock=clock,
            camera=SimulatedCamera(clock, fail_capture=bool(scenario.faults.get("camera_capture"))),
            scanner=SimulatedTagScanner(observations),
            detector=SimulatedDetector(detections),
            storage=InMemoryStorage(clock, fail_upload=bool(scenario.faults.get("storage_upload"))),
            registry=InMemoryTagRegistry(entries.values()),
            assignment_policy=assignment_policy,
        )

    def run(self) -> ScenarioRunResult:
        assignments: list[RiderAssignment] = []
        captures: list[str] = []
        uploads: list[StorageReceipt] = []
        failures: list[dict[str, str]] = []
        scanned = asyncio.run(self._start_and_scan())
        pending: dict[str, list[TagObservation]] = defaultdict(list)
        for observation in scanned:
            pending[observation.tag_id].append(observation)
        for values in pending.values():
            values.sort(key=lambda item: item.observed_at_monotonic_s)

        simulation = BehavioralSimulation(duration_s=self.scenario.duration_s, seed=self.scenario.seed)

        def on_tag(event: Event, _: BehavioralSimulation):
            self.clock.advance_to(event.at_s)
            tag_id = str(event.payload["tag_id"])
            available = pending[tag_id]
            if not available:
                failure = {"component": "tag_scanner", "error": f"no observation for {tag_id}"}
                failures.append(failure)
                return (Event(event.at_s, "component_failed", failure),)
            observation = available.pop(0)
            return (
                Event(
                    event.at_s,
                    "tag_observed",
                    {
                        "tag_id": observation.tag_id,
                        "rssi_dbm": observation.rssi_dbm,
                        "acceleration_mps2": observation.acceleration_mps2.model_dump(),
                    },
                ),
            )

        def on_detection(event: Event, _: BehavioralSimulation):
            self.clock.advance_to(event.at_s)
            frame_id = str(event.payload["frame_id"])
            frame = VideoFrame(frame_id, event.at_s, 1920, 1080, b"simulated-frame")
            detected = asyncio.run(self.detector.detect(frame))
            if not detected:
                return (Event(event.at_s, "detection_missed", {"frame_id": frame_id}),)

            decision_at_s = event.at_s + self.assignment_policy.jump_window_after_s
            return (
                Event(
                    decision_at_s,
                    "evaluate_rider_assignment",
                    {"frame_id": frame_id, "jump_at_s": event.at_s},
                ),
            )

        def on_assignment(event: Event, _: BehavioralSimulation):
            self.clock.advance_to(event.at_s)
            frame_id = str(event.payload["frame_id"])
            assignment = assign_rider(
                scanned,
                self.registry,
                assigned_at_monotonic_s=event.at_s,
                jump_at_monotonic_s=float(event.payload["jump_at_s"]),
                **self.assignment_policy.model_dump(),
            )
            assignments.append(assignment)
            generated = [
                Event(
                    event.at_s,
                    "rider_assignment",
                    assignment.model_dump(mode="json"),
                )
            ]
            request = CaptureRequest(
                request_id=f"capture-{frame_id}",
                requested_at_monotonic_s=event.at_s,
                pre_roll_s=15,
                post_roll_s=5,
                assignment=assignment,
            )
            try:
                media = asyncio.run(self.camera.capture(request))
                captures.append(media.asset.asset_id)
                generated.append(
                    Event(event.at_s, "capture_completed", {"asset_id": media.asset.asset_id})
                )
            except ComponentError as exc:
                failure = {"component": "camera", "error": str(exc)}
                failures.append(failure)
                generated.append(Event(event.at_s, "component_failed", failure))
                return tuple(generated)

            owner = assignment.rider_id or assignment.status.value
            object_key = f"{owner}/{media.asset.filename}"
            try:
                receipt = asyncio.run(self.storage.upload(media, object_key))
                uploads.append(receipt)
                generated.append(
                    Event(
                        event.at_s,
                        "clip_uploaded",
                        {"asset_id": receipt.asset_id, "object_key": receipt.object_key},
                    )
                )
            except ComponentError as exc:
                failure = {"component": "storage", "error": str(exc)}
                failures.append(failure)
                generated.append(Event(event.at_s, "component_failed", failure))
            return tuple(generated)

        simulation.subscribe("tag_enters_range", on_tag)
        simulation.subscribe("tag_observation", on_tag)
        simulation.subscribe("person_detected", on_detection)
        simulation.subscribe("evaluate_rider_assignment", on_assignment)
        for index, item in enumerate(self.scenario.timeline):
            payload = dict(item.payload)
            if item.event == "person_detected":
                payload["frame_id"] = f"frame-{index}"
            simulation.schedule(Event(item.at_s, item.event, payload))

        trace = simulation.run()
        asyncio.run(self._stop())
        return ScenarioRunResult(
            trace=trace,
            assignments=tuple(assignments),
            captures=tuple(captures),
            uploads=tuple(uploads),
            failures=tuple(failures),
        )

    async def _start_and_scan(self) -> tuple[TagObservation, ...]:
        await self.camera.connect()
        await self.camera.start_preview()
        return tuple([item async for item in self.scanner.observations()])

    async def _stop(self) -> None:
        await self.camera.stop_preview()
        await self.camera.disconnect()
