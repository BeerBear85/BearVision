import asyncio
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from bearvision.contracts import BoundingBox, PersonDetection, TagObservation, Vector3
from bearvision.domain import BearTagObservationBuffer
from bearvision.edge import BearVisionOrchestrator, EdgeLifecycleState
from bearvision.ports import (
    CaptureWindow,
    CaptureWindowBasis,
    CaptureWindowPrecision,
    ComponentUnavailable,
    PermanentComponentError,
    PreparedClip,
    ProcessingTraceEvent,
    VideoFrame,
)
from bearvision.simulation import (
    InMemoryJobQueue,
    SimulatedCamera,
    SimulatedDetector,
    SimulatedTagScanner,
    VirtualClock,
)


NOW = datetime(2026, 8, 12, tzinfo=timezone.utc)


def observation(tag_id: str, at_s: float, *, active: bool, rssi_dbm: int) -> TagObservation:
    return TagObservation(
        tag_id=tag_id,
        observed_at_utc=NOW,
        observed_at_monotonic_s=at_s,
        rssi_dbm=rssi_dbm,
        acceleration_mps2=(Vector3(x=4, y=2, z=19) if active else Vector3(x=0, y=0, z=9.80665)),
    )


def build_orchestrator() -> tuple[BearVisionOrchestrator, InMemoryJobQueue]:
    clock = VirtualClock(NOW)
    detection = PersonDetection(
        frame_id="frame-1",
        observed_at_monotonic_s=1,
        bounding_box=BoundingBox(x_px=1, y_px=1, width_px=20, height_px=40),
        confidence=0.9,
    )
    queue = InMemoryJobQueue()
    return (
        BearVisionOrchestrator(
            clock=clock,
            camera=SimulatedCamera(clock),
            scanner=SimulatedTagScanner(()),
            detector=SimulatedDetector({"frame-1": (detection,)}),
            job_queue=queue,
            edge_device_id="edge-test",
            recording_duration_s=5,
        ),
        queue,
    )


class FlakyQueue(InMemoryJobQueue):
    def __init__(self, failures: list[Exception]) -> None:
        super().__init__()
        self.failures = failures
        self.publish_attempts = 0

    async def publish(self, manifest, video, observations):
        self.publish_attempts += 1
        if self.failures:
            raise self.failures.pop(0)
        return await super().publish(manifest, video, observations)


def build_with_queue(
    queue: InMemoryJobQueue,
    *,
    max_restarts: int,
    event_sink=None,
) -> BearVisionOrchestrator:
    clock = VirtualClock(NOW)
    detection = PersonDetection(
        frame_id="frame-1",
        observed_at_monotonic_s=1,
        bounding_box=BoundingBox(x_px=1, y_px=1, width_px=20, height_px=40),
        confidence=0.9,
    )
    return BearVisionOrchestrator(
        clock=clock,
        camera=SimulatedCamera(clock),
        scanner=SimulatedTagScanner(()),
        detector=SimulatedDetector({"frame-1": (detection,)}),
        job_queue=queue,
        edge_device_id="edge-test",
        recording_duration_s=5,
        max_restarts=max_restarts,
        event_sink=event_sink,
    )


def test_orchestrator_packages_all_whole_clip_observations_without_identity() -> None:
    async def exercise() -> None:
        orchestrator, queue = build_orchestrator()
        await orchestrator.start()
        for at_s in (1.1, 2, 4, 5.9):
            orchestrator.add_tag_observation(observation("active", at_s, active=True, rssi_dbm=-65))
            orchestrator.add_tag_observation(
                observation("nearby", at_s, active=False, rssi_dbm=-40)
            )

        result = await orchestrator.process_frame(VideoFrame("frame-1", 1, 100, 100, b"pixels"))

        assert result is not None
        assert result.clip_start_monotonic_s == 1
        assert result.clip_end_monotonic_s == 6
        assert len(result.observations) == 8
        assert {item.bear_tag_id for item in result.observations} == {"active", "nearby"}
        manifest_json = queue.packages[result.request_id]["manifest.json"].decode()
        assert "rider" not in manifest_json.lower()
        assert "user" not in manifest_json.lower()
        assert result.published
        assert result.states == (
            EdgeLifecycleState.RECORDING,
            EdgeLifecycleState.PACKAGING,
            EdgeLifecycleState.UPLOADING,
            EdgeLifecycleState.MONITORING,
        )
        assert queue.snapshot()["counts"]["ready"] == 1
        await orchestrator.stop()

    asyncio.run(exercise())


def test_post_processing_adjusts_job_utc_interval_and_observation_offsets() -> None:
    async def exercise() -> None:
        class TrimmingProcessor:
            async def process(self, media):
                return PreparedClip(media=media, source_start_offset_s=1, duration_s=2)

        orchestrator, _ = build_orchestrator()
        orchestrator.clip_processor = TrimmingProcessor()
        await orchestrator.start()
        for at_s in (1.1, 2, 4, 5.9):
            orchestrator.add_tag_observation(observation("active", at_s, active=True, rssi_dbm=-60))
        result = await orchestrator.process_frame(VideoFrame("frame-1", 1, 100, 100, b"pixels"))
        assert result is not None
        assert result.manifest.capture_started_at == NOW + timedelta(seconds=1)
        assert result.manifest.capture_ended_at == NOW + timedelta(seconds=3)
        assert tuple(item.offset_ms for item in result.observations) == (0, 2000)
        assert result.job_start_monotonic_s == 2
        assert result.job_end_monotonic_s == 4
        assert result.processing is not None
        assert EdgeLifecycleState.POST_PROCESSING in result.states
        await orchestrator.stop()

    asyncio.run(exercise())


def test_hardware_style_and_recorded_cameras_share_the_complete_publish_pipeline() -> None:
    async def run_pipeline(*, estimated_camera_timing: bool):
        class Camera(SimulatedCamera):
            async def capture(self, request):
                capture = await super().capture(request)
                if not estimated_camera_timing:
                    return capture
                return replace(
                    capture,
                    actual_window=replace(
                        capture.actual_window,
                        precision=CaptureWindowPrecision.ESTIMATED,
                        basis=CaptureWindowBasis.CAMERA_COMMAND_TIMING,
                    ),
                )

        class Processor:
            def __init__(self) -> None:
                self.calls = 0

            async def process(self, media):
                self.calls += 1
                return PreparedClip(
                    media=media,
                    source_start_offset_s=1,
                    duration_s=2,
                    trace_events=(
                        ProcessingTraceEvent(
                            kind="processor_completed",
                            payload={"processor": "parity-test"},
                        ),
                    ),
                )

        class Queue(InMemoryJobQueue):
            def __init__(self) -> None:
                super().__init__()
                self.publish_calls = 0

            async def publish(self, manifest, video, observations):
                self.publish_calls += 1
                return await super().publish(manifest, video, observations)

        clock = VirtualClock(NOW)
        camera = Camera(clock)
        processor = Processor()
        queue = Queue()
        orchestrator = BearVisionOrchestrator(
            clock=clock,
            camera=camera,
            scanner=SimulatedTagScanner(()),
            detector=SimulatedDetector({}),
            job_queue=queue,
            edge_device_id="parity-edge",
            recording_duration_s=5,
            capture_pre_roll_s=15,
            clip_processor=processor,
            observation_retention_s=30,
        )
        detection = PersonDetection(
            frame_id="same-frame",
            observed_at_monotonic_s=20,
            bounding_box=BoundingBox(x_px=1, y_px=1, width_px=20, height_px=40),
            confidence=0.9,
        )
        for at_s in (5.5, 6, 7.9, 8, 24):
            orchestrator.add_tag_observation(
                observation("active", at_s, active=True, rssi_dbm=-60)
            )

        await orchestrator.start()
        clock.advance_to(20)
        result = await orchestrator.handle_detection(detection)
        await orchestrator.stop()
        return result, queue, processor

    async def exercise() -> None:
        hardware, hardware_queue, hardware_processor = await run_pipeline(
            estimated_camera_timing=True
        )
        recorded, recorded_queue, recorded_processor = await run_pipeline(
            estimated_camera_timing=False
        )

        assert hardware.manifest == recorded.manifest
        assert hardware.observations == recorded.observations
        assert hardware.job_start_monotonic_s == recorded.job_start_monotonic_s == 6
        assert hardware.job_end_monotonic_s == recorded.job_end_monotonic_s == 8
        assert hardware.events == recorded.events
        assert tuple(event.kind for event in hardware.events) == (
            "capture_started",
            "finalize_clip",
            "capture_completed",
            "processor_completed",
            "clip_uploaded",
        )
        assert hardware.published and recorded.published
        assert hardware_queue.packages == recorded_queue.packages
        assert hardware_queue.publish_calls == recorded_queue.publish_calls == 1
        assert hardware_processor.calls == recorded_processor.calls == 1

    asyncio.run(exercise())


def test_hindsight_extends_clip_and_manifest_before_detection() -> None:
    async def exercise() -> None:
        clock = VirtualClock(NOW)
        detection = PersonDetection(
            frame_id="frame-hindsight",
            observed_at_monotonic_s=20,
            bounding_box=BoundingBox(x_px=1, y_px=1, width_px=20, height_px=40),
            confidence=0.9,
        )
        camera = SimulatedCamera(clock)
        orchestrator = BearVisionOrchestrator(
            clock=clock,
            camera=camera,
            scanner=SimulatedTagScanner(()),
            detector=SimulatedDetector({}),
            job_queue=InMemoryJobQueue(),
            edge_device_id="edge-test",
            recording_duration_s=5,
            capture_pre_roll_s=15,
            observation_retention_s=30,
        )
        for at_s in (4.9, 5, 12, 25, 25.1):
            orchestrator.add_tag_observation(observation("active", at_s, active=True, rssi_dbm=-60))

        await orchestrator.start()
        clock.advance_to(20)
        result = await orchestrator.handle_detection(detection)
        assert result.clip_start_monotonic_s == 5
        assert result.clip_end_monotonic_s == 25
        assert result.manifest.capture_started_at == NOW + timedelta(seconds=5)
        assert result.manifest.capture_ended_at == NOW + timedelta(seconds=25)
        assert tuple(item.offset_ms for item in result.observations) == (0, 7000, 20000)
        await orchestrator.stop()

    asyncio.run(exercise())


def test_orchestrator_anchors_manifest_and_observations_to_actual_raw_window() -> None:
    async def exercise() -> None:
        class DelayedCamera(SimulatedCamera):
            async def capture(self, request):
                capture = await super().capture(request)
                return replace(
                    capture,
                    actual_window=CaptureWindow(
                        start_monotonic_s=(
                            capture.actual_window.start_monotonic_s + 0.25
                        ),
                        end_monotonic_s=capture.actual_window.end_monotonic_s,
                        precision=CaptureWindowPrecision.ESTIMATED,
                        basis=CaptureWindowBasis.CAMERA_COMMAND_TIMING,
                    ),
                )

        clock = VirtualClock(NOW)
        camera = DelayedCamera(clock)
        orchestrator = BearVisionOrchestrator(
            clock=clock,
            camera=camera,
            scanner=SimulatedTagScanner(()),
            detector=SimulatedDetector({}),
            job_queue=InMemoryJobQueue(),
            edge_device_id="edge-test",
            recording_duration_s=5,
            capture_pre_roll_s=15,
            observation_retention_s=30,
        )
        detection = PersonDetection(
            frame_id="delayed-camera",
            observed_at_monotonic_s=20,
            bounding_box=BoundingBox(x_px=1, y_px=1, width_px=20, height_px=40),
            confidence=0.9,
        )
        for at_s in (5.0, 5.25, 25.25, 25.5):
            orchestrator.add_tag_observation(
                observation("active", at_s, active=True, rssi_dbm=-60)
            )

        await orchestrator.start()
        clock.advance_to(20)
        result = await orchestrator.handle_detection(detection)

        assert result.raw_capture.requested_window.start_monotonic_s == 5
        assert result.clip_start_monotonic_s == 5.25
        assert result.clip_end_monotonic_s == 25
        assert result.manifest.capture_started_at == NOW + timedelta(seconds=5.25)
        assert result.manifest.capture_ended_at == NOW + timedelta(seconds=25)
        assert tuple(item.offset_ms for item in result.observations) == (0,)
        await orchestrator.stop()

    asyncio.run(exercise())


def test_repeated_and_concurrent_detections_publish_one_job() -> None:
    async def exercise() -> None:
        orchestrator, queue = build_orchestrator()
        await orchestrator.start()
        frame = VideoFrame("frame-1", 1, 100, 100, b"pixels")
        first, second = await asyncio.gather(
            orchestrator.process_frame(frame), orchestrator.process_frame(frame)
        )
        assert first == second
        assert len(queue.packages) == 1
        repeated = PersonDetection(
            frame_id="frame-2",
            observed_at_monotonic_s=2,
            bounding_box=BoundingBox(x_px=1, y_px=1, width_px=20, height_px=40),
            confidence=0.9,
        )
        assert await orchestrator.handle_detection(repeated) == first
        assert len(queue.packages) == 1
        await orchestrator.stop()

    asyncio.run(exercise())


def test_later_detection_does_not_extend_active_capture_end() -> None:
    async def exercise() -> None:
        class BlockingClock(VirtualClock):
            def __init__(self) -> None:
                super().__init__(NOW)
                self.sleeping = asyncio.Event()
                self.release = asyncio.Event()

            async def sleep(self, delay_s: float) -> None:
                self.sleeping.set()
                await self.release.wait()
                self.advance_by(delay_s)

        clock = BlockingClock()
        camera = SimulatedCamera(clock)
        queue = InMemoryJobQueue()
        orchestrator = BearVisionOrchestrator(
            clock=clock,
            camera=camera,
            scanner=SimulatedTagScanner(()),
            detector=SimulatedDetector({}),
            job_queue=queue,
            edge_device_id="edge-test",
            recording_duration_s=5,
        )
        first_detection = PersonDetection(
            frame_id="first",
            observed_at_monotonic_s=1,
            bounding_box=BoundingBox(x_px=1, y_px=1, width_px=20, height_px=40),
            confidence=0.9,
        )
        later_detection = first_detection.model_copy(
            update={"frame_id": "later", "observed_at_monotonic_s": 2}
        )

        await orchestrator.start()
        first_task = asyncio.create_task(orchestrator.handle_detection(first_detection))
        await clock.sleeping.wait()
        later_task = asyncio.create_task(orchestrator.handle_detection(later_detection))
        await asyncio.sleep(0)
        clock.release.set()
        first, later = await asyncio.gather(first_task, later_task)

        assert later == first
        assert first.raw_capture.requested_window.end_monotonic_s == 6
        assert len(queue.packages) == 1
        await orchestrator.stop()

    asyncio.run(exercise())


def test_observation_buffer_retains_bounded_ordered_clip_data() -> None:
    buffer = BearTagObservationBuffer(retention_s=2)
    buffer.append(observation("active", 1, active=True, rssi_dbm=-60))
    buffer.append(observation("active", 2, active=True, rssi_dbm=-60))
    buffer.append(observation("active", 4, active=True, rssi_dbm=-60))
    buffer.append(observation("active", 3, active=True, rssi_dbm=-60))
    assert tuple(item.observed_at_monotonic_s for item in buffer.between(2, 4)) == (2, 3, 4)


def test_retryable_queue_failure_recovers_but_permanent_failure_does_not() -> None:
    async def exercise() -> None:
        retrying = FlakyQueue([ComponentUnavailable("Box is temporarily offline")])
        orchestrator = build_with_queue(retrying, max_restarts=1)
        await orchestrator.start()
        result = await orchestrator.process_frame(VideoFrame("frame-1", 1, 100, 100, b"pixels"))
        assert result is not None and result.published
        assert retrying.publish_attempts == 2
        assert EdgeLifecycleState.RECOVERING in result.states
        await orchestrator.stop()

        permanent = FlakyQueue([PermanentComponentError("invalid credentials")])
        orchestrator = build_with_queue(permanent, max_restarts=5)
        await orchestrator.start()
        try:
            await orchestrator.process_frame(VideoFrame("frame-1", 1, 100, 100, b"pixels"))
        except PermanentComponentError:
            pass
        else:
            raise AssertionError("permanent component failure was not propagated")
        assert permanent.publish_attempts == 1
        await orchestrator.stop()

    asyncio.run(exercise())


def test_lifecycle_transitions_are_emitted_through_the_runtime_seam() -> None:
    async def exercise() -> None:
        events = []
        orchestrator = build_with_queue(
            InMemoryJobQueue(),
            max_restarts=0,
            event_sink=events.append,
        )

        await orchestrator.start()
        result = await orchestrator.process_frame(
            VideoFrame("frame-1", 1, 100, 100, b"pixels")
        )

        assert result is not None
        stages = [
            event.payload["stage"]
            for event in events
            if event.kind == "lifecycle_changed"
        ]
        assert stages == [
            "initializing",
            "monitoring",
            "recording",
            "packaging",
            "uploading",
            "monitoring",
        ]
        await orchestrator.stop()

    asyncio.run(exercise())


def test_failed_publication_is_retained_and_retried_without_recapture() -> None:
    async def exercise() -> None:
        queue = FlakyQueue([ComponentUnavailable("Box is temporarily offline")])
        events = []
        orchestrator = build_with_queue(
            queue,
            max_restarts=0,
            event_sink=events.append,
        )
        await orchestrator.start()

        result = await orchestrator.process_frame(
            VideoFrame("frame-1", 1, 100, 100, b"pixels")
        )

        assert result is not None and not result.published
        failure = next(event for event in events if event.kind == "component_failed")
        assert failure.payload["retryable"] is True
        assert queue.publish_attempts == 1
        capture_count = len(orchestrator.camera.captures)

        retry_events = await orchestrator.retry_failure(failure.payload["failure_id"])

        assert queue.publish_attempts == 2
        assert len(orchestrator.camera.captures) == capture_count
        assert [event.kind for event in retry_events] == [
            "lifecycle_changed",
            "clip_uploaded",
            "failure_resolved",
            "lifecycle_changed",
        ]
        assert queue.snapshot()["counts"]["ready"] == 1
        await orchestrator.stop()

    asyncio.run(exercise())


def test_stop_waits_for_active_capture_before_disconnect() -> None:
    async def exercise() -> None:
        class BlockingClock(VirtualClock):
            def __init__(self) -> None:
                super().__init__(NOW)
                self.sleeping = asyncio.Event()
                self.release = asyncio.Event()

            async def sleep(self, delay_s: float) -> None:
                self.sleeping.set()
                await self.release.wait()
                self.advance_by(delay_s)

        clock = BlockingClock()
        detection = PersonDetection(
            frame_id="frame-1",
            observed_at_monotonic_s=1,
            bounding_box=BoundingBox(x_px=1, y_px=1, width_px=20, height_px=40),
            confidence=0.9,
        )
        camera = SimulatedCamera(clock)
        orchestrator = BearVisionOrchestrator(
            clock=clock,
            camera=camera,
            scanner=SimulatedTagScanner(()),
            detector=SimulatedDetector({"frame-1": (detection,)}),
            job_queue=InMemoryJobQueue(),
            edge_device_id="edge-test",
            recording_duration_s=5,
        )
        await orchestrator.start()
        capture = asyncio.create_task(
            orchestrator.process_frame(VideoFrame("frame-1", 1, 100, 100, b"pixels"))
        )
        await clock.sleeping.wait()
        stopping = asyncio.create_task(orchestrator.stop())
        await asyncio.sleep(0)
        assert not stopping.done() and camera.connected
        clock.release.set()
        await capture
        await stopping
        assert not camera.connected

    asyncio.run(exercise())


def test_feature_flags_suppress_unwanted_work() -> None:
    async def exercise() -> None:
        clock = VirtualClock(NOW)
        camera = SimulatedCamera(clock)
        disabled = BearVisionOrchestrator(
            clock=clock,
            camera=camera,
            scanner=SimulatedTagScanner(()),
            detector=SimulatedDetector({}),
            job_queue=InMemoryJobQueue(),
            edge_device_id="edge-test",
            recording_duration_s=5,
            detection_enabled=False,
            preview_enabled=False,
            upload_enabled=False,
        )
        await disabled.start()
        assert not camera.previewing
        assert await disabled.process_frame(VideoFrame("ignored", 0, 10, 10, b"x")) is None
        await disabled.stop()

    asyncio.run(exercise())


def test_run_cleans_up_partially_started_hardware() -> None:
    async def exercise() -> None:
        class FailingFrameSource:
            async def open(self, preview_source: str) -> None:
                raise ComponentUnavailable(f"cannot open {preview_source}")

            async def close(self) -> None:
                raise AssertionError("a frame source that never opened must not be closed")

            async def frames(self):
                if False:
                    yield VideoFrame("unused", 0, 1, 1, b"")

        clock = VirtualClock(NOW)
        camera = SimulatedCamera(clock)
        orchestrator = BearVisionOrchestrator(
            clock=clock,
            camera=camera,
            scanner=SimulatedTagScanner(()),
            detector=SimulatedDetector({}),
            job_queue=InMemoryJobQueue(),
            edge_device_id="edge-test",
            recording_duration_s=5,
            frame_source=FailingFrameSource(),
        )

        try:
            await orchestrator.run()
        except ComponentUnavailable:
            pass
        else:
            raise AssertionError("preview startup failure was not propagated")

        assert not camera.connected
        assert not camera.previewing
        assert orchestrator.state is EdgeLifecycleState.STOPPED

    asyncio.run(exercise())


def test_run_stops_immediately_when_bear_tag_stream_fails() -> None:
    async def exercise() -> None:
        class BlockingFrameSource:
            def __init__(self) -> None:
                self.opened = False
                self.closed = False

            async def open(self, preview_source: str) -> None:
                self.opened = True

            async def close(self) -> None:
                self.closed = True

            async def frames(self):
                await asyncio.Event().wait()
                if False:
                    yield VideoFrame("unused", 0, 1, 1, b"")

        class FailingScanner:
            async def observations(self):
                if False:
                    yield observation("unused", 0, active=False, rssi_dbm=-100)
                raise ComponentUnavailable("BLE adapter stopped")

        clock = VirtualClock(NOW)
        camera = SimulatedCamera(clock)
        frame_source = BlockingFrameSource()
        orchestrator = BearVisionOrchestrator(
            clock=clock,
            camera=camera,
            scanner=FailingScanner(),
            detector=SimulatedDetector({}),
            job_queue=InMemoryJobQueue(),
            edge_device_id="edge-test",
            recording_duration_s=5,
            frame_source=frame_source,
        )

        running = asyncio.create_task(orchestrator.run())
        try:
            try:
                await asyncio.wait_for(asyncio.shield(running), timeout=0.1)
            except ComponentUnavailable as exc:
                assert "BLE adapter stopped" in str(exc)
            except TimeoutError:
                raise AssertionError("BLE failure was not supervised") from None
            else:
                raise AssertionError("BLE failure was not propagated")
        finally:
            if not running.done():
                running.cancel()
                with suppress(asyncio.CancelledError):
                    await running

        assert frame_source.closed
        assert not camera.connected
        assert orchestrator.state is EdgeLifecycleState.STOPPED

    asyncio.run(exercise())


def test_stop_attempts_every_cleanup_after_one_cleanup_fails() -> None:
    async def exercise() -> None:
        class FailingCloseFrameSource:
            async def open(self, preview_source: str) -> None:
                return None

            async def close(self) -> None:
                raise RuntimeError("frame cleanup failed")

            async def frames(self):
                if False:
                    yield VideoFrame("unused", 0, 1, 1, b"")

        clock = VirtualClock(NOW)
        camera = SimulatedCamera(clock)
        orchestrator = BearVisionOrchestrator(
            clock=clock,
            camera=camera,
            scanner=SimulatedTagScanner(()),
            detector=SimulatedDetector({}),
            job_queue=InMemoryJobQueue(),
            edge_device_id="edge-test",
            recording_duration_s=5,
            frame_source=FailingCloseFrameSource(),
        )
        await orchestrator.start()

        try:
            await orchestrator.stop()
        except RuntimeError as exc:
            assert "frame cleanup failed" in str(exc)
        else:
            raise AssertionError("cleanup failure was not propagated")

        assert not camera.connected
        assert not camera.previewing
        assert orchestrator.state is EdgeLifecycleState.STOPPED

    asyncio.run(exercise())
