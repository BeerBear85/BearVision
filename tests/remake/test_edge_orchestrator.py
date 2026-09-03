import asyncio
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

from bearvision.contracts import BoundingBox, PersonDetection, TagObservation, Vector3
from bearvision.domain import BearTagObservationBuffer
from bearvision.edge import BearVisionOrchestrator, EdgeLifecycleState, RawClipPipeline
from bearvision.ports import ComponentUnavailable, PreparedClip, VideoFrame
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
            Vector3(x=4, y=2, z=19)
            if active
            else Vector3(x=0, y=0, z=9.80665)
        ),
    )


def detection(frame_id: str, at_s: float) -> PersonDetection:
    return PersonDetection(
        frame_id=frame_id,
        observed_at_monotonic_s=at_s,
        bounding_box=BoundingBox(x_px=1, y_px=1, width_px=20, height_px=40),
        confidence=0.9,
    )


def build_orchestrator(
    tmp_path: Path,
    *,
    detector=None,
    processor=None,
    queue=None,
    event_sink=None,
    clock=None,
    camera=None,
    **options,
):
    capture_dir = tmp_path / "captures"
    capture_dir.mkdir(exist_ok=True)
    clock = clock or VirtualClock(NOW)
    camera = camera or SimulatedCamera(clock, capture_dir=capture_dir)
    queue = queue or InMemoryJobQueue()
    pipeline = RawClipPipeline(
        capture_dir=capture_dir,
        clock=clock,
        clip_processor=processor,
        job_queue=queue,
        edge_device_id="edge-test",
        upload_enabled=options.pop("upload_enabled", True),
    )
    orchestrator = BearVisionOrchestrator(
        clock=clock,
        camera=camera,
        scanner=options.pop("scanner", SimulatedTagScanner(())),
        detector=detector or SimulatedDetector({}),
        edge_device_id="edge-test",
        recording_duration_s=5,
        raw_clip_pipeline=pipeline,
        event_sink=event_sink,
        **options,
    )
    return orchestrator, queue, pipeline


def test_detection_returns_immediate_disposition_and_pipeline_packages_observations(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        person = detection("frame-1", 1)
        orchestrator, queue, pipeline = build_orchestrator(
            tmp_path,
            detector=SimulatedDetector({"frame-1": (person,)}),
        )
        await orchestrator.start()
        for at_s in (1.1, 2, 4, 5.9):
            orchestrator.add_tag_observation(
                observation("active", at_s, active=True, rssi_dbm=-65)
            )

        evaluation = await orchestrator.process_frame(
            VideoFrame("frame-1", 1, 100, 100, b"pixels")
        )

        assert evaluation.capture_disposition == "scheduled"
        await orchestrator.wait_until_idle()
        job = pipeline.snapshot().jobs[0]
        assert job.status == "completed"
        manifest_json = queue.packages[job.request_id]["manifest.json"].decode()
        assert "rider" not in manifest_json.lower()
        assert "user" not in manifest_json.lower()
        assert queue.snapshot()["counts"]["ready"] == 1
        await orchestrator.stop()

    asyncio.run(exercise())


def test_processing_and_upload_do_not_change_live_monitoring_state(tmp_path: Path) -> None:
    async def exercise() -> None:
        class Processor:
            async def process(self, media):
                return PreparedClip(media=media, source_start_offset_s=1, duration_s=2)

        events = []
        orchestrator, _, pipeline = build_orchestrator(
            tmp_path,
            processor=Processor(),
            event_sink=events.append,
        )
        await orchestrator.start()
        assert await orchestrator.handle_detection(detection("frame-1", 1)) == "scheduled"
        await orchestrator.wait_until_idle()

        assert orchestrator.state is EdgeLifecycleState.MONITORING
        assert pipeline.snapshot().jobs[0].status == "completed"
        lifecycle = [
            event.payload["stage"]
            for event in events
            if event.kind == "lifecycle_changed"
        ]
        assert lifecycle == ["initializing", "monitoring"]
        await orchestrator.stop()

    asyncio.run(exercise())


def test_episode_and_request_id_deduplication_do_not_extend_capture(tmp_path: Path) -> None:
    async def exercise() -> None:
        clock = VirtualClock(NOW)
        orchestrator, _, _ = build_orchestrator(
            tmp_path,
            clock=clock,
            detection_cooldown_s=5,
        )
        await orchestrator.start()

        first = await orchestrator.handle_detection(detection("first", 1))
        duplicate = await orchestrator.handle_detection(detection("first", 1))
        same_episode = await orchestrator.handle_detection(detection("later", 2))
        await orchestrator.wait_until_idle()

        assert (first, duplicate, same_episode) == (
            "scheduled", "duplicate", "same_episode",
        )
        capture = orchestrator.camera.captures["capture-first"]
        assert capture.requested_window.end_monotonic_s == 6
        assert len(orchestrator.camera.captures) == 1
        await orchestrator.stop()

    asyncio.run(exercise())


def test_stop_waits_for_pending_capture_before_disconnect(tmp_path: Path) -> None:
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
        capture_dir = tmp_path / "captures"
        capture_dir.mkdir()
        camera = SimulatedCamera(clock, capture_dir=capture_dir)
        orchestrator, _, _ = build_orchestrator(
            tmp_path,
            clock=clock,
            camera=camera,
        )
        await orchestrator.start()
        assert await orchestrator.handle_detection(detection("frame-1", 1)) == "scheduled"
        await clock.sleeping.wait()

        stopping = asyncio.create_task(orchestrator.stop())
        await asyncio.sleep(0)
        assert not stopping.done() and camera.connected
        clock.release.set()
        await stopping
        assert not camera.connected

    asyncio.run(exercise())


def test_feature_flags_suppress_detection_and_preview(tmp_path: Path) -> None:
    async def exercise() -> None:
        orchestrator, _, _ = build_orchestrator(
            tmp_path,
            detection_enabled=False,
            preview_enabled=False,
            upload_enabled=False,
        )
        await orchestrator.start()
        assert not orchestrator.camera.previewing
        evaluation = await orchestrator.process_frame(VideoFrame("ignored", 0, 10, 10, b"x"))
        assert evaluation.events == ()
        assert evaluation.capture_disposition is None
        await orchestrator.stop()

    asyncio.run(exercise())


def test_run_cleans_up_partially_started_hardware(tmp_path: Path) -> None:
    async def exercise() -> None:
        class FailingFrameSource:
            async def open(self, preview_source: str) -> None:
                raise ComponentUnavailable(f"cannot open {preview_source}")

            async def close(self) -> None:
                raise AssertionError("a frame source that never opened must not be closed")

            async def frames(self):
                if False:
                    yield VideoFrame("unused", 0, 1, 1, b"")

        orchestrator, _, _ = build_orchestrator(
            tmp_path,
            frame_source=FailingFrameSource(),
        )
        try:
            await orchestrator.run()
        except ComponentUnavailable:
            pass
        else:
            raise AssertionError("preview startup failure was not propagated")

        assert not orchestrator.camera.connected
        assert orchestrator.state is EdgeLifecycleState.STOPPED

    asyncio.run(exercise())


def test_run_stops_when_bear_tag_stream_fails(tmp_path: Path) -> None:
    async def exercise() -> None:
        class BlockingFrameSource:
            def __init__(self) -> None:
                self.closed = False

            async def open(self, preview_source: str) -> None:
                return None

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

        frame_source = BlockingFrameSource()
        orchestrator, _, _ = build_orchestrator(
            tmp_path,
            frame_source=frame_source,
            scanner=FailingScanner(),
        )
        running = asyncio.create_task(orchestrator.run())
        try:
            await asyncio.wait_for(asyncio.shield(running), timeout=1)
        except ComponentUnavailable as exc:
            assert "BLE adapter stopped" in str(exc)
        finally:
            if not running.done():
                running.cancel()
                with suppress(asyncio.CancelledError):
                    await running

        assert frame_source.closed
        assert not orchestrator.camera.connected

    asyncio.run(exercise())


def test_observation_buffer_retains_bounded_ordered_clip_data() -> None:
    buffer = BearTagObservationBuffer(retention_s=2)
    buffer.append(observation("active", 1, active=True, rssi_dbm=-60))
    buffer.append(observation("active", 2, active=True, rssi_dbm=-60))
    buffer.append(observation("active", 4, active=True, rssi_dbm=-60))
    buffer.append(observation("active", 3, active=True, rssi_dbm=-60))
    assert tuple(item.observed_at_monotonic_s for item in buffer.between(2, 4)) == (2, 3, 4)
