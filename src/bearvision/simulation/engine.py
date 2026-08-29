"""A small deterministic event engine for behavioural system simulations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import heapq
import itertools
import random
from typing import Any

from bearvision.contracts import RuntimeEventKind


Payload = Mapping[str, Any]


@dataclass(frozen=True)
class Event:
    """An event scheduled at an absolute virtual time."""

    at_s: float
    kind: RuntimeEventKind
    payload: Payload = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.at_s < 0:
            raise ValueError("event time must be non-negative")
        if not self.kind.strip():
            raise ValueError("event kind must not be empty")


@dataclass(frozen=True)
class TraceEntry:
    """An immutable record of a dispatched event."""

    at_s: float
    sequence: int
    kind: RuntimeEventKind
    payload: dict[str, Any]


EventHandler = Callable[[Event, "BehavioralSimulation"], Iterable[Event] | None]


class BehavioralSimulation:
    """Dispatch scheduled events using virtual time and stable ordering.

    Events at the same timestamp are processed in insertion order. Handlers may
    return more events, allowing simulated components to react without waiting
    for wall-clock time.
    """

    def __init__(self, *, duration_s: float, seed: int = 0) -> None:
        if duration_s < 0:
            raise ValueError("duration must be non-negative")
        self.duration_s = float(duration_s)
        self.seed = seed
        self.random = random.Random(seed)
        self.now_s = 0.0
        self._sequence = itertools.count()
        self._queue: list[tuple[float, int, Event]] = []
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._trace: list[TraceEntry] = []

    @property
    def trace(self) -> tuple[TraceEntry, ...]:
        """Return the trace as an immutable snapshot."""

        return tuple(self._trace)

    def subscribe(self, kind: RuntimeEventKind, handler: EventHandler) -> None:
        """Register a handler for one event kind."""

        if not kind.strip():
            raise ValueError("event kind must not be empty")
        self._handlers[kind].append(handler)

    def schedule(self, event: Event) -> None:
        """Schedule an event inside the configured simulation window."""

        if event.at_s < self.now_s:
            raise ValueError("cannot schedule an event in the past")
        if event.at_s > self.duration_s:
            raise ValueError("event is outside the simulation duration")
        sequence = next(self._sequence)
        heapq.heappush(self._queue, (event.at_s, sequence, event))

    def run(self) -> tuple[TraceEntry, ...]:
        """Run until the queue is empty, then advance to scenario end."""

        while self._queue:
            at_s, sequence, event = heapq.heappop(self._queue)
            self.now_s = at_s
            self._trace.append(
                TraceEntry(
                    at_s=at_s,
                    sequence=sequence,
                    kind=event.kind,
                    payload=dict(event.payload),
                )
            )
            for handler in tuple(self._handlers.get(event.kind, ())):
                generated = handler(event, self)
                if generated:
                    for response in generated:
                        self.schedule(response)

        self.now_s = self.duration_s
        return self.trace
