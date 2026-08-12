"""Hybrid scenario runner: recorded frames, real YOLO and simulated infrastructure."""

from __future__ import annotations

import asyncio
from pathlib import Path

from bearvision.adapters import FfmpegVideoClipper, YoloDetectorAdapter
from bearvision.config import AssignmentConfig
from bearvision.config.models import ClipExtractionConfig
from bearvision.contracts import ScenarioDefinition
from bearvision.edge.orchestrator import BearVisionOrchestrator, OrchestrationResult

from .adapters import InMemoryStorage, InMemoryTagRegistry, SimulatedTagScanner, VirtualClock
from .engine import TraceEntry
from .runner import ScenarioRunResult, evaluate_expectations
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
        storage: InMemoryStorage,
    ) -> None:
        self.scenario = scenario
        self.orchestrator = orchestrator
        self.clock = clock
        self.camera = camera
        self.frame_source = frame_source
        self.storage = storage

    @classmethod
    def from_scenario(
        cls,
        scenario: ScenarioDefinition,
        *,
        assignment_policy: AssignmentConfig | None = None,
        recording_duration_s: float = 5.0,
        repository_root: Path | None = None,
        capture_dir: Path | None = None,
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
        camera = RecordedVideoCamera(
            video_path,
            clock,
            clipper=FfmpegVideoClipper(ClipExtractionConfig()),
            capture_dir=capture_dir or root / "temp/captures",
        )
        frame_source = RecordedVideoFrameSource(sample_fps=scenario.video.sample_fps)
        storage = InMemoryStorage(clock)
        orchestrator = BearVisionOrchestrator(
            clock=clock,
            camera=camera,
            scanner=SimulatedTagScanner(()),
            detector=YoloDetectorAdapter(handler),
            storage=storage,
            registry=InMemoryTagRegistry(registry),
            assignment_policy=assignment_policy or AssignmentConfig(),
            recording_duration_s=recording_duration_s,
            observation_retention_s=max(30.0, scenario.duration_s + recording_duration_s),
            frame_source=frame_source,
            ble_logging_enabled=False,
        )
        for observation in observations:
            orchestrator.add_tag_observation(observation)
        return cls(
            scenario,
            orchestrator=orchestrator,
            clock=clock,
            camera=camera,
            frame_source=frame_source,
            storage=storage,
        )

    def run(self) -> ScenarioRunResult:
        return asyncio.run(self._run())

    async def _run(self) -> ScenarioRunResult:
        trace_events: list[tuple[float, str, dict]] = []
        results: dict[str, OrchestrationResult] = {}
        detection_times_s: list[float] = []
        failures: list[dict[str, str]] = []
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
                result = await self.orchestrator.process_frame(frame)
                if result is not None:
                    detection_times_s.append(frame.observed_at_monotonic_s)
                    trace_events.append(
                        (
                            frame.observed_at_monotonic_s,
                            "person_detected",
                            {"frame_id": frame.frame_id},
                        )
                    )
                    results.setdefault(result.request_id, result)
                    break
        except Exception as exc:
            failures.append({"component": "video_scenario", "error": str(exc)})
        finally:
            await self.orchestrator.stop()

        for result in results.values():
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
                        "rider_assignment",
                        result.assignment.model_dump(mode="json"),
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
                ]
            )
            if result.upload is not None:
                trace_events.append(
                    (
                        result.clip_end_monotonic_s,
                        "clip_uploaded",
                        {
                            "asset_id": result.upload.asset_id,
                            "object_key": result.upload.object_key,
                        },
                    )
                )

        ordered = sorted(enumerate(trace_events), key=lambda item: (item[1][0], item[0]))
        trace = tuple(
            TraceEntry(at_s=at_s, sequence=sequence, kind=kind, payload=payload)
            for sequence, (_, (at_s, kind, payload)) in enumerate(ordered)
        )
        assignments = tuple(result.assignment for result in results.values())
        captures = tuple(media.asset.asset_id for media in self.camera.captures.values())
        uploads = tuple(receipt for _, receipt in self.storage.objects.values())
        expectation_failures = evaluate_expectations(
            self.scenario,
            assignments,
            captures,
            uploads,
            detection_times_s=tuple(detection_times_s),
        )
        return ScenarioRunResult(
            trace=trace,
            assignments=assignments,
            captures=captures,
            uploads=uploads,
            failures=tuple(failures),
            expectation_failures=expectation_failures,
        )
