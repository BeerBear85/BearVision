"""BearVision 3 composition roots for real and behavioural execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Callable, TYPE_CHECKING

from bearvision.adapters import (
    BoxStorageAdapter,
    BleakKBeaconSource,
    GoProCameraAdapter,
    KBeaconTagScannerAdapter,
    OpenCvPreviewFrameSource,
    SystemClock,
    YoloDetectorAdapter,
)
from bearvision.config import AssignmentConfig, EdgeConfig
from bearvision.ports import Camera, Clock, Detector, Storage, TagRegistry, TagScanner
from bearvision.ports import FrameSource
from bearvision.contracts import ScenarioDefinition, TagRegistryEntry
from .orchestrator import BearVisionOrchestrator

if TYPE_CHECKING:
    from bearvision.simulation.runner import ClosedLoopScenarioRunner
    from bearvision.simulation.video_runner import VideoScenarioRunner


class ConfiguredTagRegistry:
    def __init__(self, entries: Iterable[TagRegistryEntry]) -> None:
        self._entries = {entry.tag_id: entry for entry in entries}

    def resolve(self, tag_id: str) -> TagRegistryEntry | None:
        entry = self._entries.get(tag_id)
        return entry if entry is not None and entry.enabled else None

    def entries(self) -> tuple[TagRegistryEntry, ...]:
        return tuple(self._entries.values())


@dataclass(frozen=True, slots=True)
class RealEdgeComponents:
    clock: Clock
    camera: Camera
    scanner: TagScanner
    detector: Detector
    storage: Storage
    registry: TagRegistry
    assignment_policy: AssignmentConfig
    frame_source: FrameSource


def build_behavioral_system(
    scenario: ScenarioDefinition,
    assignment_policy: AssignmentConfig | None = None,
    *,
    capture_dir: Path | None = None,
) -> "ClosedLoopScenarioRunner | VideoScenarioRunner":
    if scenario.components.frames == "video":
        from bearvision.simulation.video_runner import VideoScenarioRunner

        return VideoScenarioRunner.from_scenario(
            scenario,
            assignment_policy=assignment_policy,
            capture_dir=capture_dir,
        )
    supported_synthetic = (
        scenario.components.frames == "synthetic"
        and scenario.components.detector == "declared"
        and scenario.components.bear_tag == "synthetic"
        and scenario.components.camera == "simulated"
        and scenario.components.storage == "memory"
    )
    if not supported_synthetic:
        raise ValueError(
            "this component-source combination is declared but not implemented as a "
            "behavioural composition"
        )
    from bearvision.simulation.runner import ClosedLoopScenarioRunner

    return ClosedLoopScenarioRunner.from_scenario(
        scenario,
        assignment_policy=assignment_policy,
    )


def build_real_system(
    config: EdgeConfig,
    *,
    capture_dir: str | Path,
    scratch_dir: str | Path,
    gopro_factory: Callable[[], Any] | None = None,
    beacon_factory: Callable[[], Any] | None = None,
    detector_factory: Callable[[str], Any] | None = None,
    box_factory: Callable[[dict[str, dict[str, str]]], Any] | None = None,
) -> RealEdgeComponents:
    """Instantiate existing hardware implementations behind BearVision 3 ports."""

    if gopro_factory is None:
        from bearvision.integrations.gopro_controller import GoProController

        gopro_factory = GoProController
    if beacon_factory is None:
        beacon_factory = BleakKBeaconSource
    if detector_factory is None:
        from bearvision.integrations.opencv_dnn import DnnHandler

        detector_factory = DnnHandler
    if box_factory is None:
        from bearvision.integrations.box_handler import BoxHandler

        box_factory = BoxHandler

    legacy_detector = detector_factory(config.detection.model)
    legacy_detector.confidence_threshold = config.detection.confidence_threshold
    legacy_detector.init()
    box_config = {
        "STORAGE_COMMON": {
            "secret_key_name": config.storage.credential_env,
            "secret_key_name_2": config.storage.secondary_credential_env or "",
        },
        "BOX": {"root_folder": config.storage.root_folder},
    }
    clock = SystemClock()
    registry = ConfiguredTagRegistry(
        TagRegistryEntry(tag_id=item.tag_id, rider_id=item.rider_id, enabled=item.enabled)
        for item in config.tag_registry
    )
    return RealEdgeComponents(
        clock=clock,
        camera=GoProCameraAdapter(
            gopro_factory(),
            clock,
            capture_dir,
            hindsight_enabled=config.recording.hindsight_enabled,
        ),
        scanner=KBeaconTagScannerAdapter(beacon_factory(), clock),
        detector=YoloDetectorAdapter(legacy_detector),
        storage=BoxStorageAdapter(box_factory(box_config), clock, scratch_dir),
        registry=registry,
        assignment_policy=config.assignment,
        frame_source=OpenCvPreviewFrameSource(
            clock,
            max_fps=config.performance.max_fps,
            queue_size=config.performance.callback_queue_size,
            drain_old_frames=config.performance.buffer_drain,
        ),
    )


def build_real_orchestrator(
    config: EdgeConfig,
    *,
    capture_dir: str | Path,
    scratch_dir: str | Path,
    gopro_factory: Callable[[], Any] | None = None,
    beacon_factory: Callable[[], Any] | None = None,
    detector_factory: Callable[[str], Any] | None = None,
    box_factory: Callable[[dict[str, dict[str, str]]], Any] | None = None,
) -> BearVisionOrchestrator:
    """Build the production orchestrator without exposing legacy SDKs to it."""

    components = build_real_system(
        config,
        capture_dir=capture_dir,
        scratch_dir=scratch_dir,
        gopro_factory=gopro_factory,
        beacon_factory=beacon_factory,
        detector_factory=detector_factory,
        box_factory=box_factory,
    )
    return BearVisionOrchestrator(
        clock=components.clock,
        camera=components.camera,
        scanner=components.scanner,
        detector=components.detector,
        storage=components.storage,
        registry=components.registry,
        assignment_policy=components.assignment_policy,
        recording_duration_s=config.recording.post_detection_duration_s,
        observation_retention_s=max(30.0, config.recording.post_detection_duration_s),
        frame_source=components.frame_source,
        detection_enabled=config.detection.enabled,
        upload_enabled=config.features.cloud_upload,
        preview_enabled=True,
        ble_logging_enabled=config.features.ble_logging,
        detection_cooldown_s=config.detection.cooldown_s,
        max_restarts=config.error_recovery.max_restarts,
        restart_delay_s=config.error_recovery.restart_delay_s,
    )
