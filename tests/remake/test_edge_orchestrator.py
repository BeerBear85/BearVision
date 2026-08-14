import asyncio
from datetime import datetime, timedelta, timezone

from bearvision.contracts import BoundingBox, PersonDetection, TagObservation, Vector3
from bearvision.domain import BearTagObservationBuffer
from bearvision.edge import BearVisionOrchestrator, EdgeLifecycleState
from bearvision.ports import (
    ComponentUnavailable,
    PermanentComponentError,
    PreparedClip,
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
        acceleration_mps2=(
            Vector3(x=4, y=2, z=19) if active else Vector3(x=0, y=0, z=9.80665)
        ),
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


def build_with_queue(queue: InMemoryJobQueue, *, max_restarts: int) -> BearVisionOrchestrator:
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
    )


def test_orchestrator_packages_all_whole_clip_observations_without_identity() -> None:
    async def exercise() -> None:
        orchestrator, queue = build_orchestrator()
        await orchestrator.start()
        for at_s in (1.1, 2, 4, 5.9):
            orchestrator.add_tag_observation(observation("active", at_s, active=True, rssi_dbm=-65))
            orchestrator.add_tag_observation(observation("nearby", at_s, active=False, rssi_dbm=-40))

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
            orchestrator.add_tag_observation(
                observation("active", at_s, active=True, rssi_dbm=-60)
            )
        result = await orchestrator.process_frame(
            VideoFrame("frame-1", 1, 100, 100, b"pixels")
        )
        assert result is not None
        assert result.manifest.capture_started_at == NOW + timedelta(seconds=1)
        assert result.manifest.capture_ended_at == NOW + timedelta(seconds=3)
        assert tuple(item.offset_ms for item in result.observations) == (0, 2000)
        assert EdgeLifecycleState.POST_PROCESSING in result.states
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
