"""Asynchronous BearVision Edge orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
import logging
from typing import TypeVar

from bearvision.contracts import (
    BearTagJobObservation,
    CaptureRequest,
    EdgeJobManifest,
    PersonDetection,
    TagObservation,
)
from bearvision.domain import BearTagObservationBuffer
from bearvision.ports import (
    Camera,
    CapturedClip,
    CapturedMedia,
    ClipProcessor,
    Clock,
    ComponentError,
    Detector,
    FrameSource,
    InvalidComponentData,
    JobQueue,
    PermanentComponentError,
    TagScanner,
    VideoFrame,
)

from .job_package import build_edge_job


logger = logging.getLogger(__name__)
T = TypeVar("T")


class EdgeLifecycleState(StrEnum):
    INITIALIZING = "initializing"
    MONITORING = "monitoring"
    RECORDING = "recording"
    POST_PROCESSING = "post_processing"
    PACKAGING = "packaging"
    UPLOADING = "uploading"
    RECOVERING = "recovering"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    request_id: str
    clip_start_monotonic_s: float
    clip_end_monotonic_s: float
    manifest: EdgeJobManifest
    observations: tuple[BearTagJobObservation, ...]
    raw_capture: CapturedClip
    media: CapturedMedia
    published: bool
    states: tuple[EdgeLifecycleState, ...]


class BearVisionOrchestrator:
    """Capture clips and publish anonymous whole-clip evidence to the cloud queue."""

    def __init__(
        self,
        *,
        clock: Clock,
        camera: Camera,
        scanner: TagScanner,
        detector: Detector,
        job_queue: JobQueue,
        edge_device_id: str,
        recording_duration_s: float,
        capture_pre_roll_s: float = 0.0,
        clip_processor: ClipProcessor | None = None,
        observation_retention_s: float = 30.0,
        frame_source: FrameSource | None = None,
        detection_enabled: bool = True,
        upload_enabled: bool = True,
        preview_enabled: bool = True,
        ble_logging_enabled: bool = True,
        detection_cooldown_s: float = 0.0,
        max_restarts: int = 0,
        restart_delay_s: float = 0.0,
    ) -> None:
        if recording_duration_s <= 0:
            raise ValueError("recording_duration_s must be positive")
        if capture_pre_roll_s < 0:
            raise ValueError("capture_pre_roll_s must not be negative")
        if observation_retention_s < capture_pre_roll_s + recording_duration_s:
            raise ValueError("observation retention must cover the entire clip")
        if not edge_device_id:
            raise ValueError("edge_device_id must not be empty")
        self.clock = clock
        self.camera = camera
        self.scanner = scanner
        self.detector = detector
        self.job_queue = job_queue
        self.edge_device_id = edge_device_id
        self.recording_duration_s = recording_duration_s
        self.capture_pre_roll_s = capture_pre_roll_s
        self.clip_processor = clip_processor
        self.frame_source = frame_source
        self.detection_enabled = detection_enabled
        self.upload_enabled = upload_enabled
        self.preview_enabled = preview_enabled
        self.ble_logging_enabled = ble_logging_enabled
        self.detection_cooldown_s = detection_cooldown_s
        self.max_restarts = max_restarts
        self.restart_delay_s = restart_delay_s
        self.observations = BearTagObservationBuffer(observation_retention_s)
        self.state = EdgeLifecycleState.INITIALIZING
        self._tag_task: asyncio.Task[None] | None = None
        self._active_clip: asyncio.Task[OrchestrationResult] | None = None
        self._clip_lock = asyncio.Lock()
        self._results: dict[str, OrchestrationResult] = {}
        self._completed_clips: list[OrchestrationResult] = []

    async def start(self) -> None:
        await self._retry_component("connect camera", self.camera.connect, [])
        if self.preview_enabled:
            preview_source = await self._retry_component(
                "start camera preview", self.camera.start_preview, []
            )
            if self.frame_source is not None:
                frame_source = self.frame_source
                await self._retry_component(
                    "open preview frame source", lambda: frame_source.open(preview_source), []
                )
        self._tag_task = asyncio.create_task(self.consume_tag_observations())
        self.state = EdgeLifecycleState.MONITORING

    async def run(self) -> None:
        if self.frame_source is None:
            raise RuntimeError("run() requires a configured frame source")
        await self.start()
        try:
            async for frame in self.frame_source.frames():
                await self.process_frame(frame)
        finally:
            await self.stop()

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
        if self.frame_source is not None:
            await self.frame_source.close()
        if self.preview_enabled:
            await self.camera.stop_preview()
        await self.camera.disconnect()
        self.state = EdgeLifecycleState.STOPPED

    def add_tag_observation(self, observation: TagObservation) -> None:
        self.observations.append(observation)
        if self.ble_logging_enabled:
            logger.debug(
                "BearTag observation tag=%s rssi=%s acceleration=%s",
                observation.tag_id,
                observation.rssi_dbm,
                observation.acceleration_mps2,
            )

    async def consume_tag_observations(self) -> None:
        async for observation in self.scanner.observations():
            self.add_tag_observation(observation)

    async def process_frame(self, frame: VideoFrame) -> OrchestrationResult | None:
        if not self.detection_enabled:
            return None
        detections = await self.detector.detect(frame)
        return await self.handle_detection(detections[0]) if detections else None

    async def handle_detection(self, detection: PersonDetection) -> OrchestrationResult:
        request_id = f"capture-{detection.frame_id}"
        if request_id in self._results:
            return self._results[request_id]
        for completed in self._completed_clips:
            if (
                completed.clip_start_monotonic_s
                <= detection.observed_at_monotonic_s
                <= completed.clip_end_monotonic_s + self.detection_cooldown_s
            ):
                return completed
        async with self._clip_lock:
            if self._active_clip is None or self._active_clip.done():
                self._active_clip = asyncio.create_task(self._record_and_enqueue(detection))
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

    async def _record_and_enqueue(self, detection: PersonDetection) -> OrchestrationResult:
        request_id = f"capture-{detection.frame_id}"
        timing_reference_monotonic_s = max(
            self.clock.monotonic(), detection.observed_at_monotonic_s
        )
        timing_reference_utc = self.clock.utc_now()
        states = [EdgeLifecycleState.RECORDING]
        self.state = states[-1]
        request = CaptureRequest(
            request_id=request_id,
            requested_at_monotonic_s=detection.observed_at_monotonic_s,
            pre_roll_s=self.capture_pre_roll_s,
            post_roll_s=self.recording_duration_s,
        )
        try:
            raw_capture = await self._retry_component(
                "capture clip", lambda: self.camera.capture(request), states
            )
            self._validate_camera_capture(request, raw_capture)
            media = raw_capture.media
            clip_start_s = raw_capture.actual_window.start_monotonic_s
            clip_end_s = raw_capture.actual_window.end_monotonic_s
            job_start_s = clip_start_s
            job_end_s = clip_end_s
            capture_started_at = timing_reference_utc + timedelta(
                seconds=job_start_s - timing_reference_monotonic_s
            )
            capture_ended_at = timing_reference_utc + timedelta(
                seconds=job_end_s - timing_reference_monotonic_s
            )
            if self.clip_processor is not None:
                clip_processor = self.clip_processor
                states.append(EdgeLifecycleState.POST_PROCESSING)
                self.state = states[-1]
                prepared = await self._retry_component(
                    "process clip",
                    lambda: clip_processor.process(media),
                    states,
                )
                media = prepared.media
                job_start_s += prepared.source_start_offset_s
                job_end_s = job_start_s + prepared.duration_s
                capture_started_at += timedelta(seconds=prepared.source_start_offset_s)
                capture_ended_at = capture_started_at + timedelta(seconds=prepared.duration_s)
            states.append(EdgeLifecycleState.PACKAGING)
            self.state = states[-1]
            clip_observations = self.observations.between(job_start_s, job_end_s)
            manifest, observations = build_edge_job(
                job_id=request_id,
                edge_device_id=self.edge_device_id,
                created_at=self.clock.utc_now(),
                capture_started_at=capture_started_at,
                capture_ended_at=capture_ended_at,
                clip_start_monotonic_s=job_start_s,
                video=media,
                observations=clip_observations,
            )
            published = False
            if self.upload_enabled:
                states.append(EdgeLifecycleState.UPLOADING)
                self.state = states[-1]
                published = await self._retry_component(
                    "publish cloud job",
                    lambda: self.job_queue.publish(manifest, media, observations),
                    states,
                )
            states.append(EdgeLifecycleState.MONITORING)
            self.state = states[-1]
            return OrchestrationResult(
                request_id=request_id,
                clip_start_monotonic_s=clip_start_s,
                clip_end_monotonic_s=clip_end_s,
                manifest=manifest,
                observations=observations,
                raw_capture=raw_capture,
                media=media,
                published=published,
                states=tuple(states),
            )
        except Exception:
            self.state = EdgeLifecycleState.MONITORING
            raise

    @staticmethod
    def _validate_camera_capture(
        request: CaptureRequest,
        capture: CapturedClip,
    ) -> None:
        if capture.request_id != request.request_id:
            raise InvalidComponentData("camera returned a different capture request id")
        nominal_start_s = request.requested_at_monotonic_s - request.pre_roll_s
        if capture.requested_window.start_monotonic_s < nominal_start_s - 1e-9:
            raise InvalidComponentData("camera requested window exceeds requested pre-roll")
        if capture.requested_window.start_monotonic_s > request.requested_at_monotonic_s:
            raise InvalidComponentData("camera requested window starts after detection")
        expected_end_s = request.requested_at_monotonic_s + request.post_roll_s
        if abs(capture.requested_window.end_monotonic_s - expected_end_s) > 1e-9:
            raise InvalidComponentData("camera requested window has the wrong end time")

    async def _retry_component(
        self,
        operation_name: str,
        operation: Callable[[], Awaitable[T]],
        states: list[EdgeLifecycleState],
    ) -> T:
        for attempt in range(self.max_restarts + 1):
            try:
                return await operation()
            except (PermanentComponentError, InvalidComponentData):
                raise
            except ComponentError:
                if attempt >= self.max_restarts:
                    raise
                self.state = EdgeLifecycleState.RECOVERING
                states.append(self.state)
                logger.warning(
                    "%s failed; retrying (%d/%d)",
                    operation_name,
                    attempt + 1,
                    self.max_restarts,
                    exc_info=True,
                )
                await self.clock.sleep(self.restart_delay_s)
        raise AssertionError("retry loop exhausted unexpectedly")
