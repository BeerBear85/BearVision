"""Deterministic behavioural simulation primitives."""

from .adapters import (
    InMemoryStorage,
    InMemoryTagRegistry,
    SimulatedCamera,
    SimulatedDetector,
    SimulatedTagScanner,
    VirtualClock,
)
from .engine import BehavioralSimulation, Event, TraceEntry
from .runner import ClosedLoopScenarioRunner, ScenarioRunResult
from .scenario_inputs import generate_bear_tag_series

__all__ = [
    "BehavioralSimulation",
    "ClosedLoopScenarioRunner",
    "Event",
    "InMemoryStorage",
    "InMemoryTagRegistry",
    "SimulatedCamera",
    "SimulatedDetector",
    "SimulatedTagScanner",
    "ScenarioRunResult",
    "TraceEntry",
    "VirtualClock",
    "generate_bear_tag_series",
]
