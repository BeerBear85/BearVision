import asyncio
from datetime import datetime, timezone


from bearvision.config import AssignmentConfig
from bearvision.contracts import (
    BoundingBox,
    PersonDetection,
    RiderAssignmentStatus,
    TagObservation,
    TagRegistryEntry,
    Vector3,
)
from bearvision.domain import BearTagObservationBuffer
from bearvision.edge import BearVisionOrchestrator, EdgeLifecycleState
from bearvision.ports import ComponentUnavailable, PermanentComponentError, VideoFrame
from bearvision.simulation import (
    InMemoryStorage,
    InMemoryTagRegistry,
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


def build_orchestrator() -> tuple[BearVisionOrchestrator, InMemoryStorage]:
    clock = VirtualClock(NOW)
    detection = PersonDetection(
        frame_id="frame-1",
        observed_at_monotonic_s=1,
        bounding_box=BoundingBox(x_px=1, y_px=1, width_px=20, height_px=40),
        confidence=0.9,
    )
    storage = InMemoryStorage(clock)
    orchestrator = BearVisionOrchestrator(
        clock=clock,
        camera=SimulatedCamera(clock),
        scanner=SimulatedTagScanner(()),
        detector=SimulatedDetector({"frame-1": (detection,)}),
        storage=storage,
        registry=InMemoryTagRegistry(
            (
                TagRegistryEntry(tag_id="active", rider_id="rider-active"),
                TagRegistryEntry(tag_id="nearby", rider_id="rider-nearby"),
            )
        ),
        assignment_policy=AssignmentConfig(),
        recording_duration_s=5,
    )
    return orchestrator, storage


class FlakyStorage(InMemoryStorage):
    def __init__(self, clock: VirtualClock, failures: list[Exception]) -> None:
        super().__init__(clock)
        self.failures = failures
        self.upload_attempts = 0

    async def upload(self, media, object_key: str, *, overwrite: bool = False):
        self.upload_attempts += 1
        if self.failures:
            raise self.failures.pop(0)
        return await super().upload(media, object_key, overwrite=overwrite)


def build_with_storage(
    storage: InMemoryStorage,
    *,
    max_restarts: int,
) -> BearVisionOrchestrator:
    clock = storage.clock
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
        storage=storage,
        registry=InMemoryTagRegistry(()),
        assignment_policy=AssignmentConfig(),
        recording_duration_s=5,
        max_restarts=max_restarts,
    )


def test_orchestrator_uses_whole_clip_and_returns_to_monitoring() -> None:
    async def exercise() -> None:
        orchestrator, storage = build_orchestrator()
        await orchestrator.start()
        for at_s in (1.1, 2, 4, 5.9):
            orchestrator.add_tag_observation(observation("active", at_s, active=True, rssi_dbm=-65))
            orchestrator.add_tag_observation(observation("nearby", at_s, active=False, rssi_dbm=-40))

        result = await orchestrator.process_frame(VideoFrame("frame-1", 1, 100, 100, b"pixels"))

        assert result is not None
        assert result.clip_start_monotonic_s == 1
        assert result.clip_end_monotonic_s == 6
        assert result.assignment.status is RiderAssignmentStatus.ASSIGNED
        assert result.assignment.rider_id == "rider-active"
        assert result.assignment.evidence[0].observation_count == 4
        assert result.upload.object_key.startswith("rider-active/")
        assert result.states == (
            EdgeLifecycleState.RECORDING,
            EdgeLifecycleState.ASSIGNING,
            EdgeLifecycleState.UPLOADING,
            EdgeLifecycleState.MONITORING,
        )
        assert len(storage.objects) == 1
        await orchestrator.stop()

    asyncio.run(exercise())


def test_repeated_detection_joins_one_active_clip() -> None:
    async def exercise() -> None:
        orchestrator, storage = build_orchestrator()
        await orchestrator.start()
        orchestrator.add_tag_observation(observation("active", 1.1, active=True, rssi_dbm=-60))
        orchestrator.add_tag_observation(observation("active", 5.9, active=True, rssi_dbm=-60))
        frame = VideoFrame("frame-1", 1, 100, 100, b"pixels")
        first, second = await asyncio.gather(
            orchestrator.process_frame(frame),
            orchestrator.process_frame(frame),
        )
        assert first == second
        assert len(storage.objects) == 1

        repeated_frame = PersonDetection(
            frame_id="frame-2",
            observed_at_monotonic_s=2,
            bounding_box=BoundingBox(x_px=1, y_px=1, width_px=20, height_px=40),
            confidence=0.9,
        )
        repeated = await orchestrator.handle_detection(repeated_frame)
        assert repeated == first
        assert len(storage.objects) == 1
        await orchestrator.stop()

    asyncio.run(exercise())


def test_observation_buffer_retains_bounded_ordered_clip_data() -> None:
    buffer = BearTagObservationBuffer(retention_s=2)
    buffer.append(observation("active", 1, active=True, rssi_dbm=-60))
    buffer.append(observation("active", 2, active=True, rssi_dbm=-60))
    buffer.append(observation("active", 4, active=True, rssi_dbm=-60))
    assert tuple(item.observed_at_monotonic_s for item in buffer.between(2, 4)) == (2, 4)
    buffer.append(observation("active", 3, active=True, rssi_dbm=-60))
    assert tuple(item.observed_at_monotonic_s for item in buffer.between(2, 4)) == (2, 3, 4)


def test_retryable_storage_failure_recovers_and_records_state() -> None:
    async def exercise() -> None:
        clock = VirtualClock(NOW)
        storage = FlakyStorage(clock, [ComponentUnavailable("Box is temporarily offline")])
        orchestrator = build_with_storage(storage, max_restarts=1)
        await orchestrator.start()

        result = await orchestrator.process_frame(
            VideoFrame("frame-1", 1, 100, 100, b"pixels")
        )

        assert result is not None
        assert result.upload is not None
        assert storage.upload_attempts == 2
        assert EdgeLifecycleState.RECOVERING in result.states
        await orchestrator.stop()

    asyncio.run(exercise())


def test_permanent_storage_failure_is_not_retried() -> None:
    async def exercise() -> None:
        clock = VirtualClock(NOW)
        storage = FlakyStorage(clock, [PermanentComponentError("invalid credentials")])
        orchestrator = build_with_storage(storage, max_restarts=5)
        await orchestrator.start()

        try:
            await orchestrator.process_frame(VideoFrame("frame-1", 1, 100, 100, b"pixels"))
        except PermanentComponentError as exc:
            assert "invalid credentials" in str(exc)
        else:
            raise AssertionError("permanent component failure was not propagated")

        assert storage.upload_attempts == 1
        assert orchestrator.state is EdgeLifecycleState.MONITORING
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
            storage=InMemoryStorage(clock),
            registry=InMemoryTagRegistry(()),
            assignment_policy=AssignmentConfig(),
            recording_duration_s=5,
        )
        await orchestrator.start()
        capture = asyncio.create_task(
            orchestrator.process_frame(VideoFrame("frame-1", 1, 100, 100, b"pixels"))
        )
        await clock.sleeping.wait()

        stopping = asyncio.create_task(orchestrator.stop())
        await asyncio.sleep(0)
        assert not stopping.done()
        assert camera.connected is True

        clock.release.set()
        await capture
        await stopping
        assert camera.connected is False
        assert orchestrator.state is EdgeLifecycleState.STOPPED

    asyncio.run(exercise())


def test_distinct_concurrent_detections_share_the_active_clip() -> None:
    async def exercise() -> None:
        orchestrator, storage = build_orchestrator()
        await orchestrator.start()
        first = PersonDetection(
            frame_id="frame-a",
            observed_at_monotonic_s=1,
            bounding_box=BoundingBox(x_px=1, y_px=1, width_px=20, height_px=40),
            confidence=0.9,
        )
        second = first.model_copy(
            update={"frame_id": "frame-b", "observed_at_monotonic_s": 1.1}
        )

        first_result, second_result = await asyncio.gather(
            orchestrator.handle_detection(first),
            orchestrator.handle_detection(second),
        )

        assert first_result == second_result
        assert len(storage.objects) == 1
        await orchestrator.stop()

    asyncio.run(exercise())


def test_cooldown_and_feature_flags_suppress_unwanted_work() -> None:
    async def exercise() -> None:
        orchestrator, storage = build_orchestrator()
        orchestrator.detection_cooldown_s = 2
        await orchestrator.start()
        first = await orchestrator.process_frame(VideoFrame("frame-1", 1, 100, 100, b"pixels"))
        assert first is not None
        in_cooldown = PersonDetection(
            frame_id="frame-cooldown",
            observed_at_monotonic_s=7,
            bounding_box=BoundingBox(x_px=1, y_px=1, width_px=20, height_px=40),
            confidence=0.9,
        )
        assert await orchestrator.handle_detection(in_cooldown) == first
        assert len(storage.objects) == 1
        await orchestrator.stop()

        clock = VirtualClock(NOW)
        camera = SimulatedCamera(clock)
        disabled = BearVisionOrchestrator(
            clock=clock,
            camera=camera,
            scanner=SimulatedTagScanner(()),
            detector=SimulatedDetector({}),
            storage=InMemoryStorage(clock),
            registry=InMemoryTagRegistry(()),
            assignment_policy=AssignmentConfig(),
            recording_duration_s=5,
            detection_enabled=False,
            preview_enabled=False,
            upload_enabled=False,
        )
        await disabled.start()
        assert camera.previewing is False
        assert await disabled.process_frame(VideoFrame("ignored", 0, 10, 10, b"x")) is None
        await disabled.stop()

    asyncio.run(exercise())
