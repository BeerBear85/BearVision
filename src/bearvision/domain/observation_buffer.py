"""Bounded, time-ordered BearTag observation storage."""

from __future__ import annotations

from collections import deque
from threading import Lock
import logging

from bearvision.contracts import TagObservation


logger = logging.getLogger(__name__)


class BearTagObservationBuffer:
    """Keep recent observations and return immutable whole-clip snapshots."""

    def __init__(self, retention_s: float = 30.0) -> None:
        if retention_s <= 0:
            raise ValueError("retention_s must be positive")
        self.retention_s = retention_s
        self._items: deque[TagObservation] = deque()
        self._lock = Lock()
        self.dropped_late_observations = 0

    def append(self, observation: TagObservation) -> None:
        with self._lock:
            latest = self._items[-1].observed_at_monotonic_s if self._items else observation.observed_at_monotonic_s
            cutoff = latest - self.retention_s
            if observation.observed_at_monotonic_s < cutoff:
                self.dropped_late_observations += 1
                logger.warning(
                    "Dropping BearTag observation %.3fs older than buffer window",
                    observation.observed_at_monotonic_s,
                )
                return
            items = list(self._items)
            items.append(observation)
            items.sort(key=lambda item: item.observed_at_monotonic_s)
            self._items = deque(items)
            cutoff = self._items[-1].observed_at_monotonic_s - self.retention_s
            while self._items and self._items[0].observed_at_monotonic_s < cutoff:
                self._items.popleft()

    def between(self, start_s: float, end_s: float) -> tuple[TagObservation, ...]:
        if start_s < 0 or end_s < start_s:
            raise ValueError("observation interval is invalid")
        with self._lock:
            return tuple(
                item for item in self._items
                if start_s <= item.observed_at_monotonic_s <= end_s
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
