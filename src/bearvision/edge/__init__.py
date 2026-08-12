"""BearVision 3 edge-system composition."""

from .composition import (
    RealEdgeComponents,
    build_behavioral_system,
    build_real_orchestrator,
    build_real_system,
)
from .orchestrator import BearVisionOrchestrator, EdgeLifecycleState, OrchestrationResult

__all__ = [
    "BearVisionOrchestrator",
    "EdgeLifecycleState",
    "OrchestrationResult",
    "RealEdgeComponents",
    "build_behavioral_system",
    "build_real_orchestrator",
    "build_real_system",
]
