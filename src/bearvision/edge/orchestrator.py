"""Asynchronous BearVision Edge orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
import logging
from typing import Any, TypeVar

from bearvision.contracts import (
    BearTagJobObservation,
    CaptureRequest,
    EdgeJobManifest,
    PersonDetection,
    RuntimeEventKind,
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
    ComponentUnavailable,
    Detector,
    FrameSource,
    InvalidComponentData,
    JobQueue,
    PermanentComponentError,
    PreparedClip,
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
class OrchestrationEvent:
    """Stable trace evidence exposed without leaking orchestrator internals."""

    at_monotonic_s: float
    kind: RuntimeEventKind
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if self.at_monotonic_s < 0:
            raise ValueError("orchestration event time must not be negative")
        if not self.kind.strip():
            raise ValueError("orchestration event kind must not be empty")


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    request_id: str
    clip_start_monotonic_s: float
    clip_end_monotonic_s: float
    job_start_monotonic_s: float
    job_end_monotonic_s: float
    manifest: EdgeJobManifest
    observations: tuple[BearTagJobObservation, ...]
    raw_capture: CapturedClip
    media: CapturedMedia
    processing: PreparedClip | None
    published: bool
    states: tuple[EdgeLifecycleState, ...]
    events: tuple[OrchestrationEvent, ...]


@dataclass(frozen=True, slots=True)
class FrameEvaluation:
    """Detection trace and optional completed capture for one preview frame."""

    events: tuple[OrchestrationEvent, ...]
    result: OrchestrationResult | None


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
        self._camera_connected = False
        self._preview_started = False
        self._frame_source_open = False

    async def start(self) -> None:
        await self._retry_component("connect camera", self.camera.connect, [])
        self._camera_connected = True
        if self.preview_enabled:
            preview_source = await self._retry_component(
                "start camera preview", self.camera.start_preview, []
            )
            self._preview_started = True
            if self.frame_source is not None:
                frame_source = self.frame_source
                await self._retry_component(
                    "open preview frame source", lambda: frame_source.open(preview_source), []
                )
                self._frame_source_open = True
        self._tag_task = asyncio.create_task(self.consume_tag_observations())
        self.state = EdgeLifecycleState.MONITORING

    async def run(self) -> None:
        if self.frame_source is None:
            raise RuntimeError("run() requires a configured frame source")
        try:
            await self.start()
            frames = self.frame_source.frames().__aiter__()
            while True:
                try:
                    frame = await self._next_frame_or_tag_failure(frames)
                except StopAsyncIteration:
                    break
                await self.process_frame(frame)
        finally:
            await self.stop()

    async def stop(self) -> None:
        errors: list[BaseException] = []

        async def attempt(name: str, operation: Callable[[], Awaitable[Any]]) -> None:
            try:
                await operation()
            except BaseException as exc:
                logger.warning("%s failed during Edge shutdown", name, exc_info=True)
                errors.append(exc)

        if self._active_clip is not None:
            active_clip = self._active_clip
            await attempt("active clip completion", lambda: active_clip)
        if self._tag_task is not None:
            tag_task = self._tag_task
            self._tag_task = None
            tag_task.cancel()
            try:
                await tag_task
            except asyncio.CancelledError:
                pass
            except BaseException as exc:
                logger.warning(
                    "BearTag observation task failed during Edge shutdown",
                    exc_info=True,
                )
                errors.append(exc)
        if self.frame_source is not None and self._frame_source_open:
            frame_source = self.frame_source
            self._frame_source_open = False
            await attempt("preview frame source", frame_source.close)
        if self._preview_started:
            self._preview_started = False
            await attempt("camera preview", self.camera.stop_preview)
        if self._camera_connected:
            self._camera_connected = False
            await attempt("camera connection", self.camera.disconnect)
        self.state = EdgeLifecycleState.STOPPED
        if errors:
            raise errors[0]

    async def _next_frame_or_tag_failure(
        self, frames: AsyncIterator[VideoFrame]
    ) -> VideoFrame:
        frame_task: asyncio.Future[VideoFrame] = asyncio.ensure_future(anext(frames))
        tag_task = self._tag_task
        if tag_task is None:
            return await frame_task
        done, _ = await asyncio.wait(
            (frame_task, tag_task), return_when=asyncio.FIRST_COMPLETED
        )
        if tag_task in done:
            self._tag_task = None
            frame_task.cancel()
            with suppress(asyncio.CancelledError):
                await frame_task
            if tag_task.cancelled():
                raise ComponentUnavailable("BearTag observation stream was cancelled")
            failure = tag_task.exception()
            if failure is not None:
                raise failure
            raise ComponentUnavailable("BearTag observation stream ended unexpectedly")
        return await frame_task

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
        return (await self.evaluate_frame(frame)).result

    async def evaluate_frame(self, frame: VideoFrame) -> FrameEvaluation:
        """Evaluate one frame and return trace data through a stable public seam."""

        if not self.detection_enabled:
            return FrameEvaluation(events=(), result=None)
        detections = await self.detector.detect(frame)
        if not detections:
            return FrameEvaluation(events=(), result=None)
        detection = detections[0]
        result = await self.handle_detection(detection)
        event = OrchestrationEvent(
            at_monotonic_s=frame.observed_at_monotonic_s,
            kind="person_detected",
            payload={
                "frame_id": frame.frame_id,
                "confidence": detection.confidence,
                "bounding_box": detection.bounding_box.model_dump(mode="json"),
                "coordinate_space": {
                    "width_px": frame.width_px,
                    "height_px": frame.height_px,
                },
            },
        )
        return FrameEvaluation(events=(event,), result=result)

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
            prepared: PreparedClip | None = None
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
            events = self._build_result_events(
                request_id=request_id,
                raw_capture=raw_capture,
                processed_media=media,
                prepared=prepared,
                clip_start_s=clip_start_s,
                clip_end_s=clip_end_s,
                published=published,
            )
            states.append(EdgeLifecycleState.MONITORING)
            self.state = states[-1]
            return OrchestrationResult(
                request_id=request_id,
                clip_start_monotonic_s=clip_start_s,
                clip_end_monotonic_s=clip_end_s,
                job_start_monotonic_s=job_start_s,
                job_end_monotonic_s=job_end_s,
                manifest=manifest,
                observations=observations,
                raw_capture=raw_capture,
                media=media,
                processing=prepared,
                published=published,
                states=tuple(states),
                events=events,
            )
        except Exception:
            self.state = EdgeLifecycleState.MONITORING
            raise

    @staticmethod
    def _build_result_events(
        *,
        request_id: str,
        raw_capture: CapturedClip,
        processed_media: CapturedMedia,
        prepared: PreparedClip | None,
        clip_start_s: float,
        clip_end_s: float,
        published: bool,
    ) -> tuple[OrchestrationEvent, ...]:
        raw_media = raw_capture.media
        events = [
            OrchestrationEvent(
                at_monotonic_s=clip_start_s,
                kind="capture_started",
                payload={
                    "asset_id": raw_media.asset.asset_id,
                    "clip_end_s": clip_end_s,
                },
            ),
            OrchestrationEvent(
                at_monotonic_s=clip_end_s,
                kind="finalize_clip",
                payload={"request_id": request_id},
            ),
            OrchestrationEvent(
                at_monotonic_s=clip_end_s,
                kind="capture_completed",
                payload={
                    "asset_id": raw_media.asset.asset_id,
                    "filename": raw_media.asset.filename,
                    "size_bytes": raw_media.asset.size_bytes,
                    "clip_start_s": clip_start_s,
                    "clip_duration_s": clip_end_s - clip_start_s,
                },
            ),
        ]
        if prepared is not None:
            events.extend(
                OrchestrationEvent(
                    at_monotonic_s=(
                        clip_end_s
                        if item.source_offset_s is None
                        else clip_start_s + item.source_offset_s
                    ),
                    kind=item.kind,
                    payload=dict(item.payload),
                )
                for item in prepared.trace_events
            )
        if published:
            events.append(
                OrchestrationEvent(
                    at_monotonic_s=clip_end_s,
                    kind="clip_uploaded",
                    payload={
                        "asset_id": processed_media.asset.asset_id,
                        "object_key": f"input-queue/ready/{request_id}",
                    },
                )
            )
        return tuple(events)

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
