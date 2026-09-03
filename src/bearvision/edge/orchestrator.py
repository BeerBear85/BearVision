"""Asynchronous BearVision Edge orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
import logging
from typing import Any, Literal, TypeVar

from bearvision.contracts import (
    CaptureRequest,
    PersonDetection,
    RuntimeEventKind,
    TagObservation,
)
from bearvision.domain import BearTagObservationBuffer
from bearvision.ports import (
    Camera,
    CapturedClip,
    Clock,
    ComponentError,
    ComponentUnavailable,
    Detector,
    FrameSource,
    InvalidComponentData,
    PermanentComponentError,
    TagScanner,
    VideoFrame,
)

from .raw_clip_pipeline import RawClipJobContext, RawClipPipeline


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
    STOPPING = "stopping"
    FAILED = "failed"
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
class FrameEvaluation:
    """Detection trace and immediate capture scheduling outcome for one frame."""

    events: tuple[OrchestrationEvent, ...]
    capture_disposition: Literal["scheduled", "same_episode", "duplicate"] | None = None


@dataclass(frozen=True, slots=True)
class _ScheduledCapture:
    request: CaptureRequest
    timing_reference_monotonic_s: float
    timing_reference_utc: datetime


class BearVisionOrchestrator:
    """Capture clips and publish anonymous whole-clip evidence to the cloud queue."""

    def __init__(
        self,
        *,
        clock: Clock,
        camera: Camera,
        scanner: TagScanner,
        detector: Detector,
        edge_device_id: str,
        recording_duration_s: float,
        capture_pre_roll_s: float = 0.0,
        observation_retention_s: float = 30.0,
        frame_source: FrameSource | None = None,
        detection_enabled: bool = True,
        preview_enabled: bool = True,
        ble_logging_enabled: bool = True,
        detection_cooldown_s: float = 0.0,
        max_restarts: int = 0,
        restart_delay_s: float = 0.0,
        event_sink: Callable[[OrchestrationEvent], None] | None = None,
        raw_clip_pipeline: RawClipPipeline,
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
        self.edge_device_id = edge_device_id
        self.recording_duration_s = recording_duration_s
        self.capture_pre_roll_s = capture_pre_roll_s
        self.frame_source = frame_source
        self.detection_enabled = detection_enabled
        self.preview_enabled = preview_enabled
        self.ble_logging_enabled = ble_logging_enabled
        self.detection_cooldown_s = detection_cooldown_s
        self.max_restarts = max_restarts
        self.restart_delay_s = restart_delay_s
        self.event_sink = event_sink
        self.raw_clip_pipeline = raw_clip_pipeline
        self.observations = BearTagObservationBuffer(observation_retention_s)
        self.state = EdgeLifecycleState.INITIALIZING
        self._tag_task: asyncio.Task[None] | None = None
        self._camera_connected = False
        self._preview_started = False
        self._frame_source_open = False
        self._capture_requests: asyncio.Queue[_ScheduledCapture] = asyncio.Queue()
        self._camera_worker: asyncio.Task[None] | None = None
        self._accepting_detections = False
        self._scheduled_request_ids: set[str] = set()
        self._last_person_at_s: float | None = None
        if self.raw_clip_pipeline.event_sink is None:
            self.raw_clip_pipeline.event_sink = self._emit_pipeline_event

    async def start(self) -> None:
        self._transition(EdgeLifecycleState.INITIALIZING)
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
        await self.raw_clip_pipeline.start()
        self._accepting_detections = True
        self._camera_worker = asyncio.create_task(self._run_camera_worker())
        self._tag_task = asyncio.create_task(self.consume_tag_observations())
        self._transition(EdgeLifecycleState.MONITORING)

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
        if self.state is not EdgeLifecycleState.STOPPED:
            self._transition(EdgeLifecycleState.STOPPING)
        errors: list[BaseException] = []

        async def attempt(name: str, operation: Callable[[], Awaitable[Any]]) -> None:
            try:
                await operation()
            except BaseException as exc:
                logger.warning("%s failed during Edge shutdown", name, exc_info=True)
                errors.append(exc)

        self._accepting_detections = False
        camera_worker, self._camera_worker = self._camera_worker, None
        if camera_worker is not None:
            if not camera_worker.done():
                await attempt("pending camera captures", self._capture_requests.join)
                camera_worker.cancel()
            try:
                await camera_worker
            except asyncio.CancelledError:
                pass
            except BaseException as exc:
                logger.warning("camera worker failed during Edge shutdown", exc_info=True)
                errors.append(exc)
        await attempt("raw clip pipeline", self.raw_clip_pipeline.stop)
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
        self._transition(EdgeLifecycleState.STOPPED)
        if errors:
            raise errors[0]

    async def _next_frame_or_tag_failure(
        self, frames: AsyncIterator[VideoFrame]
    ) -> VideoFrame:
        frame_task: asyncio.Future[VideoFrame] = asyncio.ensure_future(anext(frames))
        tag_task = self._tag_task
        camera_worker = self._camera_worker
        pipeline_worker = self.raw_clip_pipeline.worker_task
        supervised: tuple[asyncio.Task[None], ...] = tuple(
            task
            for task in (tag_task, camera_worker, pipeline_worker)
            if task is not None
        )
        if not supervised:
            return await frame_task
        waiters: set[asyncio.Future[Any]] = {frame_task, *supervised}
        done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        failed_background = next(
            (task for task in supervised if task in done), None
        )
        if failed_background is not None:
            if failed_background is tag_task:
                self._tag_task = None
            frame_task.cancel()
            with suppress(asyncio.CancelledError):
                await frame_task
            component = (
                "BearTag observation stream"
                if failed_background is tag_task
                else "camera worker"
                if failed_background is camera_worker
                else "raw clip pipeline worker"
            )
            if failed_background.cancelled():
                raise ComponentUnavailable(f"{component} was cancelled")
            failure = failed_background.exception()
            if failure is not None:
                raise failure
            raise ComponentUnavailable(f"{component} ended unexpectedly")
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

    async def process_frame(self, frame: VideoFrame) -> FrameEvaluation:
        return await self.evaluate_frame(frame)

    async def evaluate_frame(self, frame: VideoFrame) -> FrameEvaluation:
        """Evaluate one frame and return trace data through a stable public seam."""

        if not self.detection_enabled:
            return FrameEvaluation(events=())
        detections = await self.detector.detect(frame)
        if not detections:
            return FrameEvaluation(events=())
        self.raw_clip_pipeline.raise_if_failed()
        self._raise_camera_worker_failure()
        events = tuple(
            self._person_detection_event(frame, detection) for detection in detections
        )
        for event in events:
            self._emit_event(event)
        disposition = await self.handle_detection(detections[0])
        return FrameEvaluation(events=events, capture_disposition=disposition)

    async def handle_detection(
        self, detection: PersonDetection
    ) -> Literal["scheduled", "same_episode", "duplicate"]:
        return await self._schedule_detection(detection)

    @staticmethod
    def _person_detection_event(
        frame: VideoFrame, detection: PersonDetection
    ) -> OrchestrationEvent:
        return OrchestrationEvent(
            at_monotonic_s=frame.observed_at_monotonic_s,
            kind="person_detected",
            payload={
                "frame_id": detection.frame_id,
                "confidence": detection.confidence,
                "bounding_box": detection.bounding_box.model_dump(mode="json"),
                "coordinate_space": {
                    "width_px": frame.width_px,
                    "height_px": frame.height_px,
                },
            },
        )

    async def _schedule_detection(
        self, detection: PersonDetection
    ) -> Literal["scheduled", "same_episode", "duplicate"]:
        self._raise_camera_worker_failure()
        if not self._accepting_detections:
            raise RuntimeError("orchestrator is not accepting capture requests")
        request_id = f"capture-{detection.frame_id}"
        if request_id in self._scheduled_request_ids:
            return "duplicate"
        previous_person_at = self._last_person_at_s
        self._last_person_at_s = detection.observed_at_monotonic_s
        if (
            previous_person_at is not None
            and detection.observed_at_monotonic_s - previous_person_at
            < self.detection_cooldown_s
        ):
            self._scheduled_request_ids.add(request_id)
            return "same_episode"
        request = CaptureRequest(
            request_id=request_id,
            requested_at_monotonic_s=detection.observed_at_monotonic_s,
            pre_roll_s=self.capture_pre_roll_s,
            post_roll_s=self.recording_duration_s,
        )
        self._scheduled_request_ids.add(request_id)
        await self._capture_requests.put(
            _ScheduledCapture(
                request=request,
                timing_reference_monotonic_s=max(
                    self.clock.monotonic(), detection.observed_at_monotonic_s
                ),
                timing_reference_utc=self.clock.utc_now(),
            )
        )
        self._emit_capture_activity("idle", None)
        return "scheduled"

    async def _run_camera_worker(self) -> None:
        while True:
            scheduled = await self._capture_requests.get()
            request = scheduled.request
            try:
                self._emit_capture_activity("capturing", request.request_id)
                raw_capture = await self._retry_component(
                    "capture clip", lambda: self.camera.capture(request), []
                )
                self._validate_camera_capture(request, raw_capture)
                clip_start_s = raw_capture.actual_window.start_monotonic_s
                clip_end_s = raw_capture.actual_window.end_monotonic_s
                capture_started_at = scheduled.timing_reference_utc + timedelta(
                    seconds=clip_start_s - scheduled.timing_reference_monotonic_s
                )
                capture_ended_at = scheduled.timing_reference_utc + timedelta(
                    seconds=clip_end_s - scheduled.timing_reference_monotonic_s
                )
                observations = self.observations.between(clip_start_s, clip_end_s)
                raw_media = raw_capture.media
                self._emit_event(
                    OrchestrationEvent(
                        at_monotonic_s=clip_start_s,
                        kind="capture_started",
                        payload={
                            "asset_id": raw_media.asset.asset_id,
                            "clip_end_s": clip_end_s,
                            "operation_id": request.request_id,
                        },
                    )
                )
                self._emit_event(
                    OrchestrationEvent(
                        at_monotonic_s=clip_end_s,
                        kind="capture_completed",
                        payload={
                            "asset_id": raw_media.asset.asset_id,
                            "filename": raw_media.asset.filename,
                            "size_bytes": raw_media.asset.size_bytes,
                            "clip_start_s": clip_start_s,
                            "clip_duration_s": clip_end_s - clip_start_s,
                            "operation_id": request.request_id,
                        },
                    )
                )
                pipeline = self.raw_clip_pipeline
                assert pipeline is not None
                await pipeline.submit(
                    raw_capture,
                    RawClipJobContext(
                        capture_started_at_utc=capture_started_at,
                        capture_ended_at_utc=capture_ended_at,
                        observations=observations,
                    ),
                )
                self._emit_event(
                    OrchestrationEvent(
                        at_monotonic_s=clip_end_s,
                        kind="finalize_clip",
                        payload={"request_id": request.request_id},
                    )
                )
            finally:
                self._capture_requests.task_done()
                self._emit_capture_activity("idle", None)

    def _emit_capture_activity(
        self,
        activity: Literal["idle", "capturing"],
        request_id: str | None,
    ) -> None:
        self._emit_event(
            OrchestrationEvent(
                at_monotonic_s=self.clock.monotonic(),
                kind="capture_activity_changed",
                payload={
                    "activity": activity,
                    "request_id": request_id,
                    "pending_captures": self._capture_requests.qsize(),
                },
            )
        )

    def _raise_camera_worker_failure(self) -> None:
        worker = self._camera_worker
        if worker is None or not worker.done() or worker.cancelled():
            return
        failure = worker.exception()
        if failure is not None:
            raise failure

    async def wait_until_captures_idle(self) -> None:
        """Wait for scheduled camera work; intended for tests and simulations."""

        self._raise_camera_worker_failure()
        await self._capture_requests.join()
        self._raise_camera_worker_failure()

    async def wait_until_idle(self) -> None:
        """Wait for camera and disk queues; never used between hardware frames."""

        await self.wait_until_captures_idle()
        await self.raw_clip_pipeline.wait_until_idle()

    def _emit_pipeline_event(
        self,
        kind: RuntimeEventKind,
        payload: dict[str, Any],
        at_monotonic_s: float,
    ) -> None:
        self._emit_event(OrchestrationEvent(at_monotonic_s, kind, payload))

    async def retry_failure(self, failure_id: str) -> tuple[OrchestrationEvent, ...]:
        """Retry one retained idempotent operation without repeating capture work."""

        await self.raw_clip_pipeline.retry(failure_id)
        return ()

    def _transition(
        self,
        state: EdgeLifecycleState,
        operation_id: str | None = None,
    ) -> OrchestrationEvent:
        self.state = state
        event = OrchestrationEvent(
            at_monotonic_s=self.clock.monotonic(),
            kind="lifecycle_changed",
            payload={"stage": state.value, "operation_id": operation_id},
        )
        self._emit_event(event)
        return event

    def _emit_event(self, event: OrchestrationEvent) -> None:
        if self.event_sink is not None:
            self.event_sink(event)

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
        if not (
            capture.actual_window.start_monotonic_s
            <= request.requested_at_monotonic_s
            <= capture.actual_window.end_monotonic_s
        ):
            raise InvalidComponentData("camera actual window does not contain detection")

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
