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
    TagRegistryEntry,
    StorageReceipt,
    Vector3,
)
from .scenario import ScenarioDefinition, TimelineEvent, load_scenario

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
    "TagObservation",
    "TagRegistryEntry",
    "StorageReceipt",
    "TimelineEvent",
    "Vector3",
    "load_scenario",
]
