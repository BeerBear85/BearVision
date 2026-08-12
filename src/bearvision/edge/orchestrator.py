"""Asynchronous BearVision 3 edge orchestration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum

from bearvision.config import AssignmentConfig
from bearvision.contracts import (
    CaptureRequest,
    PersonDetection,
    RiderAssignment,
    StorageReceipt,
    TagObservation,
)
from bearvision.domain import BearTagObservationBuffer, assign_rider
from bearvision.ports import Camera, CapturedMedia, Clock, Detector, Storage, TagRegistry, TagScanner, VideoFrame


class EdgeLifecycleState(StrEnum):
    INITIALIZING = "initializing"
    MONITORING = "monitoring"
    RECORDING = "recording"
    ASSIGNING = "assigning"
    UPLOADING = "uploading"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    request_id: str
    clip_start_monotonic_s: float
    clip_end_monotonic_s: float
    assignment: RiderAssignment
    media: CapturedMedia
    upload: StorageReceipt
    states: tuple[EdgeLifecycleState, ...]


class BearVisionOrchestrator:
    """Coordinate detection, whole-clip BearTag evidence, capture and upload."""

    def __init__(
        self,
        *,
        clock: Clock,
        camera: Camera,
        scanner: TagScanner,
        detector: Detector,
        storage: Storage,
        registry: TagRegistry,
        assignment_policy: AssignmentConfig,
        recording_duration_s: float,
        observation_retention_s: float = 30.0,
    ) -> None:
        if recording_duration_s <= 0:
            raise ValueError("recording_duration_s must be positive")
        if observation_retention_s < recording_duration_s:
            raise ValueError("observation retention must cover the entire clip")
        self.clock = clock
        self.camera = camera
        self.scanner = scanner
        self.detector = detector
        self.storage = storage
        self.registry = registry
        self.assignment_policy = assignment_policy
        self.recording_duration_s = recording_duration_s
        self.observations = BearTagObservationBuffer(observation_retention_s)
        self.state = EdgeLifecycleState.INITIALIZING
        self._tag_task: asyncio.Task[None] | None = None
        self._active_clip: asyncio.Task[OrchestrationResult] | None = None
        self._clip_lock = asyncio.Lock()
        self._results: dict[str, OrchestrationResult] = {}
        self._completed_clips: list[OrchestrationResult] = []

    async def start(self) -> None:
        await self.camera.connect()
        await self.camera.start_preview()
        self._tag_task = asyncio.create_task(self.consume_tag_observations())
        self.state = EdgeLifecycleState.MONITORING

    async def stop(self) -> None:
        if self._active_clip is not None:
            await self._active_clip
        if self._tag_task is not None:
            self._tag_task.cancel()
            try:
                await self._tag_task
            except asyncio.CancelledError:
                pass
            self._tag_task = None
        await self.camera.stop_preview()
        await self.camera.disconnect()
        self.state = EdgeLifecycleState.STOPPED

    def add_tag_observation(self, observation: TagObservation) -> None:
        """Add an observation directly, useful for callbacks and deterministic tests."""

        self.observations.append(observation)

    async def consume_tag_observations(self) -> None:
        async for observation in self.scanner.observations():
            self.add_tag_observation(observation)

    async def process_frame(self, frame: VideoFrame) -> OrchestrationResult | None:
        detections = await self.detector.detect(frame)
        if not detections:
            return None
        return await self.handle_detection(detections[0])

    async def handle_detection(self, detection: PersonDetection) -> OrchestrationResult:
        """Start one clip or join the currently active clip for repeated detections."""

        request_id = f"capture-{detection.frame_id}"
        if request_id in self._results:
            return self._results[request_id]
        for completed in self._completed_clips:
            if completed.clip_start_monotonic_s <= detection.observed_at_monotonic_s <= completed.clip_end_monotonic_s:
                return completed
        async with self._clip_lock:
            if self._active_clip is None or self._active_clip.done():
                self._active_clip = asyncio.create_task(self._record_assign_upload(detection))
            task = self._active_clip
        try:
            result = await task
        finally:
            async with self._clip_lock:
                if self._active_clip is task:
                    self._active_clip = None
        self._results[result.request_id] = result
        if result not in self._completed_clips:
            self._completed_clips.append(result)
        return result

    async def _record_assign_upload(self, detection: PersonDetection) -> OrchestrationResult:
        request_id = f"capture-{detection.frame_id}"
        clip_start_s = detection.observed_at_monotonic_s
        clip_end_s = clip_start_s + self.recording_duration_s
        states = [EdgeLifecycleState.RECORDING]
        self.state = states[-1]
        request = CaptureRequest(
            request_id=request_id,
            requested_at_monotonic_s=clip_start_s,
            pre_roll_s=0,
            post_roll_s=self.recording_duration_s,
        )

        media_task = asyncio.create_task(self.camera.capture(request))
        await self.clock.sleep(self.recording_duration_s)
        media = await media_task

        states.append(EdgeLifecycleState.ASSIGNING)
        self.state = states[-1]
        assignment = assign_rider(
            self.observations.between(clip_start_s, clip_end_s),
            self.registry,
            assigned_at_monotonic_s=max(self.clock.monotonic(), clip_end_s),
            clip_start_monotonic_s=clip_start_s,
            clip_end_monotonic_s=clip_end_s,
            **self.assignment_policy.model_dump(),
        )

        states.append(EdgeLifecycleState.UPLOADING)
        self.state = states[-1]
        owner = assignment.rider_id or assignment.status.value
        upload = await self.storage.upload(media, f"{owner}/{media.asset.filename}")
        states.append(EdgeLifecycleState.MONITORING)
        self.state = states[-1]
        return OrchestrationResult(
            request_id=request_id,
            clip_start_monotonic_s=clip_start_s,
            clip_end_monotonic_s=clip_end_s,
            assignment=assignment,
            media=media,
            upload=upload,
            states=tuple(states),
        )
