"""Hybrid scenario runner: recorded frames, real YOLO and simulated infrastructure."""

from __future__ import annotations

import asyncio
from datetime import timedelta
import hashlib
from pathlib import Path

from bearvision.adapters import FfmpegVideoClipper, YoloDetectorAdapter
from bearvision.config import AssignmentConfig
from bearvision.config.models import ClipExtractionConfig
from bearvision.contracts import ScenarioDefinition, StorageReceipt
from bearvision.edge.job_package import build_edge_job
from bearvision.edge.orchestrator import BearVisionOrchestrator, OrchestrationResult
from bearvision.processing import VirtualCameramanProcessor
from bearvision.ports import JobQueue

from bearvision.server import (
    BearTagAssignment,
    BearTagRecord,
    InMemoryUserRegistry,
    RegistryData,
    ServerWorker,
    UserRecord,
)
from .adapters import InMemoryJobQueue, SimulatedTagScanner, VirtualClock
from .engine import TraceEntry
from .runner import (
    ScenarioRunResult,
    _scenario_email,
    _scenario_user_id,
    evaluate_expectations,
)
from .scenario_inputs import generate_bear_tag_series
from .video import RecordedVideoCamera, RecordedVideoFrameSource


class VideoScenarioRunner:
    """Exercise production orchestration with media frames and the production detector."""

    def __init__(
        self,
        scenario: ScenarioDefinition,
        *,
        orchestrator: BearVisionOrchestrator,
        clock: VirtualClock,
        camera: RecordedVideoCamera,
        frame_source: RecordedVideoFrameSource,
        queue: JobQueue,
        worker: ServerWorker | None,
        post_processor: VirtualCameramanProcessor,
    ) -> None:
        self.scenario = scenario
        self.orchestrator = orchestrator
        self.clock = clock
        self.camera = camera
        self.frame_source = frame_source
        self.queue = queue
        self.worker = worker
        self.post_processor = post_processor

    @classmethod
    def from_scenario(
        cls,
        scenario: ScenarioDefinition,
        *,
        assignment_policy: AssignmentConfig | None = None,
        recording_duration_s: float = 5.0,
        repository_root: Path | None = None,
        capture_dir: Path | None = None,
        job_queue: JobQueue | None = None,
        process_server: bool = True,
    ) -> "VideoScenarioRunner":
        if scenario.video is None:
            raise ValueError("video scenario requires video configuration")
        if scenario.components.detector != "yolo":
            raise ValueError("recorded video currently requires the YOLO detector")
        if scenario.components.bear_tag != "synthetic":
            raise ValueError("video regression currently requires synthetic BearTag data")
        if scenario.components.camera != "recorded_video":
            raise ValueError("video regression currently requires camera=recorded_video")
        if scenario.components.storage != "memory":
            raise ValueError("video regression currently requires storage=memory")

        root = repository_root or Path(__file__).resolve().parents[3]
        video_path = (root / scenario.video.path).resolve()
        if root.resolve() not in video_path.parents:
            raise ValueError("scenario video must stay inside the repository")

        clock = VirtualClock()
        observations, registry = generate_bear_tag_series(scenario.synthetic_bear_tags, clock)
        from bearvision.integrations.opencv_dnn import DnnHandler

        handler = DnnHandler(scenario.detector.model)
        handler.confidence_threshold = scenario.detector.confidence_threshold
        handler.init()
        clipper = FfmpegVideoClipper(ClipExtractionConfig())
        camera = RecordedVideoCamera(
            video_path,
            clock,
            clipper=clipper,
            capture_dir=capture_dir or root / "temp/captures",
        )
        frame_source = RecordedVideoFrameSource(sample_fps=scenario.video.sample_fps)
        queue: JobQueue = job_queue or InMemoryJobQueue()
        users = tuple(
            UserRecord(
                id=_scenario_user_id(item.rider_id),
                email=_scenario_email(item.rider_id),
                displayName=item.rider_id,
            )
            for item in registry
        )
        server_registry = InMemoryUserRegistry(
            RegistryData(
                users=users,
                bearTags=tuple(BearTagRecord(id=item.tag_id) for item in registry),
                assignments=tuple(
                    BearTagAssignment(
                        id=f"scenario-{item.tag_id}",
                        userId=_scenario_user_id(item.rider_id),
                        bearTagId=item.tag_id,
                        validFrom=clock.start_utc - timedelta(days=1),
                        validTo=clock.start_utc + timedelta(days=1),
                    )
                    for item in registry
                ),
            )
        )
        orchestrator = BearVisionOrchestrator(
            clock=clock,
            camera=camera,
            scanner=SimulatedTagScanner(()),
            detector=YoloDetectorAdapter(handler),
            job_queue=queue,
            edge_device_id="scenario-video-edge",
            recording_duration_s=recording_duration_s,
            observation_retention_s=max(30.0, scenario.duration_s + recording_duration_s),
            frame_source=frame_source,
            ble_logging_enabled=False,
            # The recorded-video scenario uploads only after the virtual
            # cameraman has produced the smaller processed clip.
            upload_enabled=False,
        )
        post_handler = DnnHandler(scenario.detector.model)
        post_handler.confidence_threshold = min(
            0.25, scenario.detector.confidence_threshold
        )
        post_handler.init()
        post_processor = VirtualCameramanProcessor(
            YoloDetectorAdapter(post_handler),
            clock,
            ffmpeg_path=clipper.ffmpeg_path,
        )
        for observation in observations:
            orchestrator.add_tag_observation(observation)
        return cls(
            scenario,
            orchestrator=orchestrator,
            clock=clock,
            camera=camera,
            frame_source=frame_source,
            queue=queue,
            worker=(
                ServerWorker(queue, server_registry, clock, assignment_policy)
                if process_server
                else None
            ),
            post_processor=post_processor,
        )

    def run(self) -> ScenarioRunResult:
        return asyncio.run(self._run())

    async def _run(self) -> ScenarioRunResult:
        trace_events: list[tuple[float, str, dict]] = []
        results: dict[str, OrchestrationResult] = {}
        detection_times_s: list[float] = []
        failures: list[dict[str, str]] = []
        receipts: list[StorageReceipt] = []
        assignments = []
        await self.orchestrator.start()
        try:
            async for frame in self.frame_source.frames():
                trace_events.append(
                    (
                        frame.observed_at_monotonic_s,
                        "preview_frame",
                        {"frame_id": frame.frame_id},
                    )
                )
                if self.clock.monotonic() < frame.observed_at_monotonic_s:
                    self.clock.advance_to(frame.observed_at_monotonic_s)
                detections = await self.orchestrator.detector.detect(frame)
                if detections:
                    detection = detections[0]
                    result = await self.orchestrator.handle_detection(detection)
                    detection_times_s.append(frame.observed_at_monotonic_s)
                    trace_events.append(
                        (
                            frame.observed_at_monotonic_s,
                            "person_detected",
                            {
                                "frame_id": frame.frame_id,
                                "confidence": detection.confidence,
                                "bounding_box": detection.bounding_box.model_dump(mode="json"),
                                "coordinate_space": {
                                    "width_px": frame.width_px,
                                    "height_px": frame.height_px,
                                },
                            },
                        )
                    )
                    results.setdefault(result.request_id, result)
                    break
        except Exception as exc:
            failures.append({"component": "video_scenario", "error": str(exc)})
        finally:
            await self.orchestrator.stop()

        for result in results.values():
            processed = await self.post_processor.process(
                result.media,
                result.media.local_path.parent if result.media.local_path else Path("temp/captures"),
            )
            assert processed.media.local_path is not None
            processed_content = processed.media.local_path.read_bytes()
            adjusted_start_s = (
                result.clip_start_monotonic_s + processed.length_adjustment.source_start_s
            )
            adjusted_end_s = adjusted_start_s + processed.length_adjustment.output_duration_s
            manifest, packaged_observations = build_edge_job(
                job_id=result.request_id,
                edge_device_id=result.manifest.edge_device_id,
                created_at=result.manifest.created_at,
                capture_started_at=(
                    result.manifest.capture_started_at
                    + timedelta(seconds=processed.length_adjustment.source_start_s)
                ),
                capture_ended_at=(
                    result.manifest.capture_started_at
                    + timedelta(seconds=processed.length_adjustment.source_end_s)
                ),
                clip_start_monotonic_s=adjusted_start_s,
                video=processed.media,
                observations=self.orchestrator.observations.between(
                    adjusted_start_s, adjusted_end_s
                ),
            )
            assert manifest.video.sha256 == hashlib.sha256(processed_content).hexdigest()
            await self.queue.publish(manifest, processed.media, packaged_observations)
            server_result = await self.worker.run_once() if self.worker is not None else None
            if server_result is not None:
                assignments.append(server_result)
            receipt = StorageReceipt(
                asset_id=processed.media.asset.asset_id,
                object_key=f"input-queue/ready/{manifest.job_id}",
                stored_at_utc=manifest.created_at,
                checksum_sha256=manifest.video.sha256,
            )
            receipts.append(receipt)
            trace_events.extend(
                [
                    (
                        result.clip_start_monotonic_s,
                        "capture_started",
                        {
                            "asset_id": result.media.asset.asset_id,
                            "clip_end_s": result.clip_end_monotonic_s,
                        },
                    ),
                    (
                        result.clip_end_monotonic_s,
                        "finalize_clip",
                        {"request_id": result.request_id},
                    ),
                    (
                        result.clip_end_monotonic_s,
                        "capture_completed",
                        {
                            "asset_id": result.media.asset.asset_id,
                            "filename": result.media.asset.filename,
                            "size_bytes": result.media.asset.size_bytes,
                            "clip_start_s": result.clip_start_monotonic_s,
                            "clip_duration_s": (
                                result.clip_end_monotonic_s
                                - result.clip_start_monotonic_s
                            ),
                        },
                    ),
                    (
                        result.clip_end_monotonic_s,
                        "virtual_cameraman_completed",
                        {
                            "source_filename": result.media.asset.filename,
                            "processed_filename": processed.media.asset.filename,
                            "tracking_filename": processed.metadata_path.name,
                            "debug_video_filename": processed.debug_video_path.name,
                            "source_size_bytes": processed.source_size_bytes,
                            "processed_size_bytes": processed.processed_size_bytes,
                            "size_reduction_ratio": processed.reduction_ratio,
                            "output_width_px": self.post_processor.config.output_width_px,
                            "output_height_px": self.post_processor.config.output_height_px,
                            "state_estimator": "kalman_rts_smoother",
                            "camera_path": "zero_phase_butterworth",
                            "length_adjustment": processed.length_adjustment.to_dict(),
                        },
                    ),
                    (
                        result.clip_end_monotonic_s,
                        "clip_uploaded",
                        {
                            "asset_id": receipt.asset_id,
                            "object_key": receipt.object_key,
                        },
                    ),
                ]
            )
            if server_result is not None:
                trace_events.append(
                    (
                        result.clip_end_monotonic_s,
                        "server_assignment",
                        server_result.model_dump(mode="json", by_alias=True),
                    )
                )
            for tracking_frame in processed.tracking_frames:
                trace_events.append(
                    (
                        result.clip_start_monotonic_s + tracking_frame.source_at_s,
                        "tracking_observation",
                        {
                            **tracking_frame.to_dict(),
                            "coordinate_space": {
                                "width_px": processed.source_width_px,
                                "height_px": processed.source_height_px,
                            },
                        },
                    )
                )

        ordered = sorted(enumerate(trace_events), key=lambda item: (item[1][0], item[0]))
        trace = tuple(
            TraceEntry(at_s=at_s, sequence=sequence, kind=kind, payload=payload)
            for sequence, (_, (at_s, kind, payload)) in enumerate(ordered)
        )
        assignment_results = tuple(assignments)
        captures = tuple(media.asset.asset_id for media in self.camera.captures.values())
        uploads = tuple(receipts)
        expectation_failures = evaluate_expectations(
            self.scenario,
            assignment_results,
            captures,
            uploads,
            detection_times_s=tuple(detection_times_s),
            evaluate_server=self.worker is not None,
        )
        return ScenarioRunResult(
            trace=trace,
            assignments=assignment_results,
            captures=captures,
            uploads=uploads,
            failures=tuple(failures),
            expectation_failures=expectation_failures,
        )
