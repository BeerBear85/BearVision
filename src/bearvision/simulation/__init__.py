"""Deterministic behavioural simulation primitives."""

from .adapters import (
    InMemoryJobQueue,
    InMemoryStorage,
    InMemoryTagRegistry,
    SimulatedCamera,
    SimulatedDetector,
    SimulatedTagScanner,
    VirtualClock,
)
from .engine import BehavioralSimulation, Event, TraceEntry
from .gopro import SimulatedGoProController
from .runner import ClosedLoopScenarioRunner, ScenarioRunResult
from .scenario_inputs import generate_bear_tag_series

__all__ = [
    "BehavioralSimulation",
    "ClosedLoopScenarioRunner",
    "Event",
    "InMemoryJobQueue",
    "InMemoryStorage",
    "InMemoryTagRegistry",
    "SimulatedCamera",
    "SimulatedDetector",
    "SimulatedGoProController",
    "SimulatedTagScanner",
    "ScenarioRunResult",
    "TraceEntry",
    "VirtualClock",
    "generate_bear_tag_series",
]
