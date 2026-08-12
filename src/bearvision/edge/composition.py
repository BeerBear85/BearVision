"""BearVision 3 composition roots for real and behavioural execution."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path

from bearvision.adapters import (
    BoxStorageAdapter,
    GoProCameraAdapter,
    KBeaconTagScannerAdapter,
    SystemClock,
    YoloDetectorAdapter,
)
from bearvision.config import AssignmentConfig, EdgeConfig
from bearvision.ports import Camera, Clock, Detector, Storage, TagRegistry, TagScanner
from bearvision.simulation import ClosedLoopScenarioRunner
from bearvision.contracts import ScenarioDefinition, TagRegistryEntry
from bearvision.simulation import InMemoryTagRegistry


@dataclass(frozen=True, slots=True)
class RealEdgeComponents:
    clock: Clock
    camera: Camera
    scanner: TagScanner
    detector: Detector
    storage: Storage
    registry: TagRegistry
    assignment_policy: AssignmentConfig


def build_behavioral_system(
    scenario: ScenarioDefinition,
    assignment_policy: AssignmentConfig | None = None,
) -> ClosedLoopScenarioRunner:
    return ClosedLoopScenarioRunner.from_scenario(
        scenario,
        assignment_policy=assignment_policy,
    )


def build_real_system(
    config: EdgeConfig,
    *,
    capture_dir: str | Path,
    scratch_dir: str | Path,
) -> RealEdgeComponents:
    """Instantiate existing hardware implementations behind BearVision 3 ports."""

    gopro_type = importlib.import_module("GoProController").GoProController
    beacon_type = importlib.import_module("ble_beacon_handler").BleBeaconHandler
    detector_type = importlib.import_module("DnnHandler").DnnHandler
    box_type = importlib.import_module("BoxHandler").BoxHandler

    legacy_detector = detector_type(config.detection.model)
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
    registry = InMemoryTagRegistry(
        TagRegistryEntry(tag_id=item.tag_id, rider_id=item.rider_id, enabled=item.enabled)
        for item in config.tag_registry
    )
    return RealEdgeComponents(
        clock=clock,
        camera=GoProCameraAdapter(gopro_type(), clock, capture_dir),
        scanner=KBeaconTagScannerAdapter(beacon_type(), clock),
        detector=YoloDetectorAdapter(legacy_detector),
        storage=BoxStorageAdapter(box_type(box_config), clock, scratch_dir),
        registry=registry,
        assignment_policy=config.assignment,
    )
