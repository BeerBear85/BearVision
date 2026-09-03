"""BearVision 3 edge-system composition."""

from .composition import (
    RealEdgeComponents,
    build_real_orchestrator,
    build_real_system,
)
from .orchestrator import (
    BearVisionOrchestrator,
    EdgeLifecycleState,
    FrameEvaluation,
    OrchestrationEvent,
)
from .raw_clip_pipeline import (
    RawClipJobContext,
    RawClipJobSummary,
    RawClipPipeline,
    RawClipQueueSnapshot,
)

__all__ = [
    "BearVisionOrchestrator",
    "EdgeLifecycleState",
    "FrameEvaluation",
    "OrchestrationEvent",
    "RealEdgeComponents",
    "RawClipJobContext",
    "RawClipJobSummary",
    "RawClipPipeline",
    "RawClipQueueSnapshot",
    "build_real_orchestrator",
    "build_real_system",
]
