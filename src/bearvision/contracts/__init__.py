"""Versioned contracts shared by the core, adapters and simulations."""

from .models import (
    BoundingBox,
    CaptureRequest,
    CaptureResult,
    CaptureStatus,
    MediaAsset,
    PersonDetection,
    RiderAssignment,
    RiderAssignmentStatus,
    TagObservation,
    TagAssignmentEvidence,
    TagRegistryEntry,
    StorageReceipt,
    Vector3,
)
from .scenario import (
    ScenarioDefinition,
    ScenarioExpectation,
    ScenarioFaults,
    TimelineEvent,
    load_scenario,
)

__all__ = [
    "BoundingBox",
    "CaptureRequest",
    "CaptureResult",
    "CaptureStatus",
    "MediaAsset",
    "PersonDetection",
    "RiderAssignment",
    "RiderAssignmentStatus",
    "ScenarioDefinition",
    "ScenarioExpectation",
    "ScenarioFaults",
    "TagObservation",
    "TagAssignmentEvidence",
    "TagRegistryEntry",
    "StorageReceipt",
    "TimelineEvent",
    "Vector3",
    "load_scenario",
]
