import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from bearvision.contracts import BoundingBox, MediaAsset, PersonDetection
from bearvision.edge import BearVisionOrchestrator
from bearvision.edge.raw_clip_pipeline import RawClipJobContext, RawClipPipeline
from bearvision.ports import (
    CapturedClip,
    CapturedMedia,
    CaptureWindow,
    CaptureWindowBasis,
    CaptureWindowPrecision,
    PreparedClip,
    VideoFrame,
)
from bearvision.simulation import InMemoryJobQueue, VirtualClock
from bearvision.simulation import SimulatedDetector, SimulatedTagScanner


NOW = datetime(2026, 9, 3, tzinfo=timezone.utc)


def captured_clip(
    path: Path,
    request_id: str = "capture-frame-1",
    *,
    start_s: float = 10,
    end_s: float = 15,
) -> CapturedClip:
    content = b"raw-video"
    path.write_bytes(content)
    window = CaptureWindow(
        start_monotonic_s=start_s,
        end_monotonic_s=end_s,
        precision=CaptureWindowPrecision.EXACT,
        basis=CaptureWindowBasis.SIMULATED_MEDIA_TIMELINE,
    )
    return CapturedClip(
        request_id=request_id,
        media=CapturedMedia(
            asset=MediaAsset(
                asset_id=f"asset-{request_id}",
                filename=path.name,
                content_type="video/mp4",
                size_bytes=len(content),
                created_at_utc=NOW + timedelta(seconds=15),
            ),
            local_path=path,
        ),
        requested_window=window,
        actual_window=window,
    )


def test_submit_persists_only_metadata_and_keeps_raw_video_in_place(tmp_path: Path) -> None:
    async def exercise() -> None:
        capture_dir = tmp_path / "captures"
        capture_dir.mkdir()
        raw_path = capture_dir / "raw.mp4"
        clip = captured_clip(raw_path)
        original_identity = raw_path.stat().st_ino
        pipeline = RawClipPipeline(
            capture_dir=capture_dir,
            clock=VirtualClock(NOW),
            clip_processor=None,
            job_queue=InMemoryJobQueue(),
            edge_device_id="edge-test",
            upload_enabled=False,
        )
        await pipeline.start()

        summary = await pipeline.submit(
            clip,
            RawClipJobContext(
                capture_started_at_utc=NOW + timedelta(seconds=10),
                capture_ended_at_utc=NOW + timedelta(seconds=15),
                observations=(),
            ),
        )

        assert summary.job_id == clip.request_id
        assert summary.status == "queued"
        assert raw_path.stat().st_ino == original_identity
        assert list(capture_dir.glob("*.mp4")) == [raw_path]
        metadata_path = capture_dir / ".raw-clip-queue/queued/capture-frame-1.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["raw_clip_job_schema_version"] == "1.0"
        assert metadata["raw_filename"] == "raw.mp4"
        assert metadata["raw_media"]["asset_id"] == "asset-capture-frame-1"
        assert not list((capture_dir / ".raw-clip-queue").rglob("*.mp4"))
        await pipeline.stop()

    asyncio.run(exercise())


def test_one_worker_processes_jobs_in_persisted_fifo_order(tmp_path: Path) -> None:
    async def exercise() -> None:
        class RecordingProcessor:
            def __init__(self) -> None:
                self.filenames: list[str] = []

            async def process(self, media: CapturedMedia) -> PreparedClip:
                self.filenames.append(media.asset.filename)
                return PreparedClip(
                    media=media,
                    source_start_offset_s=0,
                    duration_s=5,
                )

        capture_dir = tmp_path / "captures"
        capture_dir.mkdir()
        processor = RecordingProcessor()
        pipeline = RawClipPipeline(
            capture_dir=capture_dir,
            clock=VirtualClock(NOW),
            clip_processor=processor,
            job_queue=InMemoryJobQueue(),
            edge_device_id="edge-test",
            upload_enabled=False,
        )
        await pipeline.start()
        context = RawClipJobContext(
            capture_started_at_utc=NOW + timedelta(seconds=10),
            capture_ended_at_utc=NOW + timedelta(seconds=15),
            observations=(),
        )

        await pipeline.submit(captured_clip(capture_dir / "one.mp4", "capture-one"), context)
        await pipeline.submit(captured_clip(capture_dir / "two.mp4", "capture-two"), context)
        await pipeline.wait_until_idle()

        assert processor.filenames == ["one.mp4", "two.mp4"]
        assert pipeline.snapshot().counts == {
            "queued": 0,
            "processing": 0,
            "failed": 0,
            "completed": 2,
        }
        await pipeline.stop()

    asyncio.run(exercise())


def test_job_failure_is_persisted_and_worker_continues_with_next_job(tmp_path: Path) -> None:
    async def exercise() -> None:
        class FailingFirstProcessor:
            async def process(self, media: CapturedMedia) -> PreparedClip:
                if media.asset.filename == "one.mp4":
                    raise RuntimeError("processing exploded")
                return PreparedClip(media=media, source_start_offset_s=0, duration_s=5)

        capture_dir = tmp_path / "captures"
        capture_dir.mkdir()
        pipeline = RawClipPipeline(
            capture_dir=capture_dir,
            clock=VirtualClock(NOW),
            clip_processor=FailingFirstProcessor(),
            job_queue=InMemoryJobQueue(),
            edge_device_id="edge-test",
            upload_enabled=False,
        )
        await pipeline.start()
        context = RawClipJobContext(
            capture_started_at_utc=NOW + timedelta(seconds=10),
            capture_ended_at_utc=NOW + timedelta(seconds=15),
            observations=(),
        )
        await pipeline.submit(captured_clip(capture_dir / "one.mp4", "capture-one"), context)
        await pipeline.submit(captured_clip(capture_dir / "two.mp4", "capture-two"), context)

        await asyncio.wait_for(pipeline.wait_until_idle(), timeout=1)

        snapshot = pipeline.snapshot()
        assert snapshot.counts["failed"] == 1
        assert snapshot.counts["completed"] == 1
        failed = json.loads(
            (capture_dir / ".raw-clip-queue/failed/capture-one.json").read_text(
                encoding="utf-8"
            )
        )
        assert failed["failed_step"] == "processing"
        assert failed["latest_failure_id"] == "failure-capture-one-processing-1"
        assert failed["technical_error"] == "processing exploded"
        assert failed["retry_checkpoint"] == "processing"
        await pipeline.stop()

    asyncio.run(exercise())


def test_restart_recovers_interrupted_processing_without_recapturing(tmp_path: Path) -> None:
    async def exercise() -> None:
        class BlockingProcessor:
            def __init__(self) -> None:
                self.started = asyncio.Event()

            async def process(self, media: CapturedMedia) -> PreparedClip:
                self.started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

        class CompletingProcessor:
            async def process(self, media: CapturedMedia) -> PreparedClip:
                return PreparedClip(media=media, source_start_offset_s=0, duration_s=5)

        capture_dir = tmp_path / "captures"
        capture_dir.mkdir()
        blocker = BlockingProcessor()
        first = RawClipPipeline(
            capture_dir=capture_dir,
            clock=VirtualClock(NOW),
            clip_processor=blocker,
            job_queue=InMemoryJobQueue(),
            edge_device_id="edge-test",
            upload_enabled=False,
        )
        await first.start()
        await first.submit(
            captured_clip(capture_dir / "raw.mp4"),
            RawClipJobContext(
                capture_started_at_utc=NOW + timedelta(seconds=10),
                capture_ended_at_utc=NOW + timedelta(seconds=15),
                observations=(),
            ),
        )
        await blocker.started.wait()
        await first.stop()
        assert (capture_dir / ".raw-clip-queue/processing/capture-frame-1.json").is_file()

        restarted = RawClipPipeline(
            capture_dir=capture_dir,
            clock=VirtualClock(NOW),
            clip_processor=CompletingProcessor(),
            job_queue=InMemoryJobQueue(),
            edge_device_id="edge-test",
            upload_enabled=False,
        )
        await restarted.start()
        await restarted.wait_until_idle()

        completed = json.loads(
            (capture_dir / ".raw-clip-queue/completed/capture-frame-1.json").read_text(
                encoding="utf-8"
            )
        )
        assert completed["processing_attempts"] == 2
        assert not (capture_dir / ".raw-clip-queue/processing/capture-frame-1.json").exists()
        await restarted.stop()

    asyncio.run(exercise())


def test_upload_retry_reuses_checkpoint_without_reprocessing(tmp_path: Path) -> None:
    async def exercise() -> None:
        class CountingProcessor:
            def __init__(self) -> None:
                self.calls = 0

            async def process(self, media: CapturedMedia) -> PreparedClip:
                self.calls += 1
                return PreparedClip(media=media, source_start_offset_s=0, duration_s=5)

        class FlakyQueue(InMemoryJobQueue):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            async def publish(self, manifest, video, observations):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("Box offline")
                return await super().publish(manifest, video, observations)

        capture_dir = tmp_path / "captures"
        capture_dir.mkdir()
        processor = CountingProcessor()
        queue = FlakyQueue()
        pipeline = RawClipPipeline(
            capture_dir=capture_dir,
            clock=VirtualClock(NOW),
            clip_processor=processor,
            job_queue=queue,
            edge_device_id="edge-test",
        )
        await pipeline.start()
        await pipeline.submit(
            captured_clip(capture_dir / "raw.mp4"),
            RawClipJobContext(
                capture_started_at_utc=NOW + timedelta(seconds=10),
                capture_ended_at_utc=NOW + timedelta(seconds=15),
                observations=(),
            ),
        )
        await pipeline.wait_until_idle()
        failed = pipeline.snapshot().jobs[0]
        assert failed.status == "failed"
        assert failed.failure_id == "failure-capture-frame-1-uploading-1"

        await pipeline.retry(failed.failure_id)
        await pipeline.wait_until_idle()

        assert processor.calls == 1
        assert queue.calls == 2
        assert pipeline.snapshot().counts["completed"] == 1
        await pipeline.stop()

    asyncio.run(exercise())


def test_preview_detection_schedules_new_episodes_while_processing_is_blocked(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        class DiskCamera:
            def __init__(self, capture_dir: Path) -> None:
                self.capture_dir = capture_dir
                self.captures: list[str] = []

            async def connect(self) -> None:
                return None

            async def disconnect(self) -> None:
                return None

            async def start_preview(self) -> str:
                return "sim://preview"

            async def stop_preview(self) -> None:
                return None

            async def capture(self, request) -> CapturedClip:
                self.captures.append(request.request_id)
                path = self.capture_dir / f"{request.request_id}.mp4"
                return captured_clip(
                    path,
                    request.request_id,
                    start_s=max(0, request.requested_at_monotonic_s - request.pre_roll_s),
                    end_s=request.requested_at_monotonic_s + request.post_roll_s,
                )

        class BlockingProcessor:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def process(self, media: CapturedMedia) -> PreparedClip:
                if not self.started.is_set():
                    self.started.set()
                    await self.release.wait()
                return PreparedClip(media=media, source_start_offset_s=0, duration_s=5)

        def detection(frame_id: str, at_s: float) -> PersonDetection:
            return PersonDetection(
                frame_id=frame_id,
                observed_at_monotonic_s=at_s,
                bounding_box=BoundingBox(x_px=1, y_px=1, width_px=20, height_px=40),
                confidence=0.9,
            )

        capture_dir = tmp_path / "captures"
        capture_dir.mkdir()
        clock = VirtualClock(NOW)
        camera = DiskCamera(capture_dir)
        processor = BlockingProcessor()
        pipeline = RawClipPipeline(
            capture_dir=capture_dir,
            clock=clock,
            clip_processor=processor,
            job_queue=InMemoryJobQueue(),
            edge_device_id="edge-test",
            upload_enabled=False,
        )
        detections = {
            "first": (detection("first", 1),),
            "same": (detection("same", 2),),
            "second": (detection("second", 7),),
        }
        orchestrator = BearVisionOrchestrator(
            clock=clock,
            camera=camera,
            scanner=SimulatedTagScanner(()),
            detector=SimulatedDetector(detections),
            edge_device_id="edge-test",
            recording_duration_s=5,
            detection_cooldown_s=5,
            raw_clip_pipeline=pipeline,
        )
        await orchestrator.start()

        clock.advance_to(1)
        first = await orchestrator.evaluate_frame(VideoFrame("first", 1, 100, 100, b"x"))
        await processor.started.wait()
        clock.advance_to(2)
        same = await orchestrator.evaluate_frame(VideoFrame("same", 2, 100, 100, b"x"))
        clock.advance_to(7)
        second = await orchestrator.evaluate_frame(VideoFrame("second", 7, 100, 100, b"x"))
        await asyncio.wait_for(orchestrator.wait_until_captures_idle(), timeout=1)

        assert first.capture_disposition == "scheduled"
        assert same.capture_disposition == "same_episode"
        assert second.capture_disposition == "scheduled"
        assert camera.captures == ["capture-first", "capture-second"]
        assert pipeline.snapshot().counts["queued"] == 1
        processor.release.set()
        await orchestrator.wait_until_idle()
        await orchestrator.stop()

    asyncio.run(exercise())


def test_missing_raw_file_is_recovered_as_retryable_job_failure(tmp_path: Path) -> None:
    async def exercise() -> None:
        class BlockingProcessor:
            def __init__(self) -> None:
                self.started = asyncio.Event()

            async def process(self, media: CapturedMedia) -> PreparedClip:
                self.started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

        capture_dir = tmp_path / "captures"
        capture_dir.mkdir()
        processor = BlockingProcessor()
        first = RawClipPipeline(
            capture_dir=capture_dir,
            clock=VirtualClock(NOW),
            clip_processor=processor,
            job_queue=InMemoryJobQueue(),
            edge_device_id="edge-test",
            upload_enabled=False,
        )
        await first.start()
        raw_path = capture_dir / "raw.mp4"
        await first.submit(
            captured_clip(raw_path),
            RawClipJobContext(
                capture_started_at_utc=NOW + timedelta(seconds=10),
                capture_ended_at_utc=NOW + timedelta(seconds=15),
                observations=(),
            ),
        )
        await processor.started.wait()
        await first.stop()
        raw_path.unlink()

        events = []
        restarted = RawClipPipeline(
            capture_dir=capture_dir,
            clock=VirtualClock(NOW),
            clip_processor=None,
            job_queue=InMemoryJobQueue(),
            edge_device_id="edge-test",
            upload_enabled=False,
            event_sink=lambda kind, payload, at_s: events.append((kind, payload, at_s)),
        )
        await restarted.start()

        failed = restarted.snapshot().jobs[0]
        assert failed.status == "failed"
        assert failed.failure_id == "failure-capture-frame-1-validation"
        component_failure = next(
            payload for kind, payload, _ in events if kind == "component_failed"
        )
        assert component_failure["scope"] == "clip_job"
        assert component_failure["job_id"] == "capture-frame-1"
        assert component_failure["retryable"] is True
        await restarted.stop()

    asyncio.run(exercise())


def test_failed_directory_prevents_retry_after_interrupted_metadata_commit(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        class BlockingProcessor:
            def __init__(self) -> None:
                self.started = asyncio.Event()

            async def process(self, media: CapturedMedia) -> PreparedClip:
                self.started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

        class RecordingProcessor:
            def __init__(self) -> None:
                self.calls = 0

            async def process(self, media: CapturedMedia) -> PreparedClip:
                self.calls += 1
                return PreparedClip(media=media, source_start_offset_s=0, duration_s=5)

        capture_dir = tmp_path / "captures"
        capture_dir.mkdir()
        blocker = BlockingProcessor()
        first = RawClipPipeline(
            capture_dir=capture_dir,
            clock=VirtualClock(NOW),
            clip_processor=blocker,
            job_queue=InMemoryJobQueue(),
            edge_device_id="edge-test",
            upload_enabled=False,
        )
        await first.start()
        await first.submit(
            captured_clip(capture_dir / "raw.mp4"),
            RawClipJobContext(
                capture_started_at_utc=NOW + timedelta(seconds=10),
                capture_ended_at_utc=NOW + timedelta(seconds=15),
                observations=(),
            ),
        )
        await blocker.started.wait()
        await first.stop()

        processing = capture_dir / ".raw-clip-queue/processing/capture-frame-1.json"
        failed = capture_dir / ".raw-clip-queue/failed/capture-frame-1.json"
        processing.replace(failed)
        assert json.loads(failed.read_text(encoding="utf-8"))["status"] == "processing"

        processor = RecordingProcessor()
        restarted = RawClipPipeline(
            capture_dir=capture_dir,
            clock=VirtualClock(NOW),
            clip_processor=processor,
            job_queue=InMemoryJobQueue(),
            edge_device_id="edge-test",
            upload_enabled=False,
        )
        await restarted.start()

        snapshot = restarted.snapshot()
        assert snapshot.counts["failed"] == 1
        assert snapshot.counts["queued"] == 0
        assert processor.calls == 0
        assert snapshot.jobs[0].failure_id == "failure-capture-frame-1-processing-1"
        await restarted.stop()

    asyncio.run(exercise())
