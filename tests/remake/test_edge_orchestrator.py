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
from bearvision.ports import VideoFrame
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
