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
]
