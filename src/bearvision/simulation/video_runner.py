"""Hybrid scenario runner: recorded frames, real YOLO and GoPro emulation."""

from __future__ import annotations

import asyncio
from pathlib import Path

from bearvision.adapters import FfmpegVideoClipper, GoProCameraAdapter, YoloDetectorAdapter
from bearvision.config import AssignmentConfig, EdgeConfig, VirtualCameramanConfig
from bearvision.config.models import ClipExtractionConfig
from bearvision.contracts import ScenarioDefinition
from bearvision.edge.orchestrator import BearVisionOrchestrator, OrchestrationResult
from bearvision.processing import VirtualCameramanJobProcessor, VirtualCameramanProcessor
from bearvision.ports import JobQueue

from bearvision.server import ServerWorker
from .adapters import InMemoryJobQueue, SimulatedTagScanner, VirtualClock
from .scenario_runtime import (
    ScenarioRunResult,
    TraceEvent,
    build_scenario_worker,
    finalize_scenario_run,
)
from .scenario_inputs import generate_bear_tag_series
from .gopro import SimulatedGoProController
from .video import RecordedVideoFrameSource


class VideoScenarioRunner:
    """Exercise production orchestration with media frames and the production detector."""

    def __init__(
        self,
        scenario: ScenarioDefinition,
        *,
        orchestrator: BearVisionOrchestrator,
        clock: VirtualClock,
        camera: GoProCameraAdapter,
        frame_source: RecordedVideoFrameSource,
        worker: ServerWorker | None,
    ) -> None:
        self.scenario = scenario
        self.orchestrator = orchestrator
        self.clock = clock
        self.camera = camera
        self.frame_source = frame_source
        self.worker = worker

    @classmethod
    def from_scenario(
        cls,
        scenario: ScenarioDefinition,
        *,
        assignment_policy: AssignmentConfig | None = None,
        edge_config: EdgeConfig | None = None,
        recording_duration_s: float | None = None,
        repository_root: Path | None = None,
        capture_dir: Path | None = None,
        job_queue: JobQueue | None = None,
        process_server: bool = True,
    ) -> "VideoScenarioRunner":
        if scenario.video is None:
            raise ValueError("video scenario requires video configuration")
        if scenario.components.detector != "yolo":
            raise ValueError("video regression requires the YOLO detector")
        if scenario.components.bear_tag != "synthetic":
            raise ValueError("video regression currently requires synthetic BearTag data")
        if scenario.components.camera != "simulated_gopro":
            raise ValueError("video regression requires camera=simulated_gopro")
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
        clipper = FfmpegVideoClipper(
            edge_config.clip_extraction if edge_config else ClipExtractionConfig()
        )
        capture_root = (capture_dir or root / "temp/captures").resolve()
        hindsight_enabled = (
            edge_config.recording.hindsight_enabled if edge_config else True
        )
        hindsight_duration_s = (
            edge_config.recording.hindsight_duration_s if edge_config else 15
        )
        recording_duration_s = (
            recording_duration_s
            if recording_duration_s is not None
            else (
                edge_config.recording.post_detection_duration_s
                if edge_config
                else 5.0
            )
        )
        controller = SimulatedGoProController(
            root_dir=capture_root / ".simulated-gopro-sd",
            preview_source=video_path,
            clock=clock,
            clipper=clipper,
        )
        camera = GoProCameraAdapter(
            controller,
            clock,
            capture_root,
            hindsight_enabled=hindsight_enabled,
            hindsight_duration_s=hindsight_duration_s,
        )
        frame_source = RecordedVideoFrameSource(sample_fps=scenario.video.sample_fps)
        queue: JobQueue = job_queue or InMemoryJobQueue()
        post_handler = DnnHandler(scenario.detector.model)
        post_handler.confidence_threshold = min(
            0.25, scenario.detector.confidence_threshold
        )
        post_handler.init()
        post_processor = VirtualCameramanProcessor(
            YoloDetectorAdapter(post_handler),
            clock,
            config=(
                edge_config.virtual_cameraman
                if edge_config
                else VirtualCameramanConfig()
            ),
            ffmpeg_path=clipper.ffmpeg_path,
        )
        orchestrator = BearVisionOrchestrator(
            clock=clock,
            camera=camera,
            scanner=SimulatedTagScanner(()),
            detector=YoloDetectorAdapter(handler),
            job_queue=queue,
            edge_device_id="scenario-video-edge",
            recording_duration_s=recording_duration_s,
            capture_pre_roll_s=(hindsight_duration_s if hindsight_enabled else 0),
            clip_processor=VirtualCameramanJobProcessor(post_processor, capture_root),
            observation_retention_s=max(
                30.0,
                scenario.duration_s
                + recording_duration_s
                + (hindsight_duration_s if hindsight_enabled else 0),
            ),
            frame_source=frame_source,
            ble_logging_enabled=False,
            upload_enabled=True,
        )
        for observation in observations:
            orchestrator.add_tag_observation(observation)
        return cls(
            scenario,
            orchestrator=orchestrator,
            clock=clock,
            camera=camera,
            frame_source=frame_source,
            worker=build_scenario_worker(
                entries=registry,
                queue=queue,
                clock=clock,
                assignment_policy=assignment_policy,
                enabled=process_server,
            ),
        )

    def run(self) -> ScenarioRunResult:
        return asyncio.run(self._run())

    async def _run(self) -> ScenarioRunResult:
        trace_events: list[TraceEvent] = []
        results: dict[str, OrchestrationResult] = {}
        detection_times_s: list[float] = []
        failures: list[dict[str, str]] = []
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
                evaluation = await self.orchestrator.evaluate_frame(frame)
                if evaluation.events:
                    detection_times_s.append(frame.observed_at_monotonic_s)
                    trace_events.extend(
                        (event.at_monotonic_s, event.kind, event.payload)
                        for event in evaluation.events
                    )
                if evaluation.result is not None:
                    results.setdefault(evaluation.result.request_id, evaluation.result)
        except Exception as exc:
            failures.append({"component": "video_scenario", "error": str(exc)})
        finally:
            await self.orchestrator.stop()

        for result in results.values():
            server_result = await self.worker.run_once() if self.worker is not None else None
            if server_result is not None:
                assignments.append(server_result)
            trace_events.extend(
                (event.at_monotonic_s, event.kind, event.payload)
                for event in result.events
            )
            if server_result is not None:
                trace_events.append(
                    (
                        result.clip_end_monotonic_s,
                        "server_assignment",
                        server_result.model_dump(mode="json", by_alias=True),
                    )
                )

        captures = tuple(
            capture.media.asset.asset_id for capture in self.camera.captures.values()
        )
        return finalize_scenario_run(
            scenario=self.scenario,
            trace_events=trace_events,
            assignments=assignments,
            captures=captures,
            edge_results=results.values(),
            failures=failures,
            detection_times_s=tuple(detection_times_s),
            evaluate_server=self.worker is not None,
        )
