"""Deterministic behavioural simulation primitives."""

from .adapters import (
    InMemoryJobQueue,
    InMemoryStorage,
    SimulatedCamera,
    SimulatedDetector,
    SimulatedTagScanner,
    VirtualClock,
)
from .engine import BehavioralSimulation, Event, TraceEntry
from .composition import build_behavioral_system
from .execution import ReplayEvent, ReplayOptions, ScenarioExecution
from .gopro import SimulatedGoProController
from .runner import ClosedLoopScenarioRunner
from .scenario_runtime import ScenarioRunResult
from .scenario_inputs import generate_bear_tag_series

__all__ = [
    "BehavioralSimulation",
    "build_behavioral_system",
    "ClosedLoopScenarioRunner",
    "Event",
    "InMemoryJobQueue",
    "InMemoryStorage",
    "SimulatedCamera",
    "SimulatedDetector",
    "SimulatedGoProController",
    "SimulatedTagScanner",
    "ScenarioRunResult",
    "ReplayEvent",
    "ReplayOptions",
    "ScenarioExecution",
    "TraceEntry",
    "VirtualClock",
    "generate_bear_tag_series",
]
