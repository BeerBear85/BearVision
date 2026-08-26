from datetime import datetime, timezone

from bearvision.contracts import TagRegistryEntry
from bearvision.simulation import InMemoryJobQueue, VirtualClock
from bearvision.simulation.scenario_runtime import (
    build_scenario_registry,
    build_scenario_worker,
    order_trace_events,
    scenario_user_id,
)


NOW = datetime(2026, 8, 26, tzinfo=timezone.utc)


def test_registry_assembly_is_deterministic_across_scenario_sources() -> None:
    clock = VirtualClock(NOW)
    entries = (
        TagRegistryEntry(tag_id="tag-b", rider_id="rider-b"),
        TagRegistryEntry(tag_id="tag-a", rider_id="rider-a"),
    )

    registry = build_scenario_registry(entries, clock).load()

    assert tuple(item.id for item in registry.bear_tags) == ("tag-a", "tag-b")
    assert tuple(item.id for item in registry.users) == (
        scenario_user_id("rider-a"),
        scenario_user_id("rider-b"),
    )
    assert tuple(item.bear_tag_id for item in registry.assignments) == (
        "tag-a",
        "tag-b",
    )


def test_disabled_server_worker_does_not_consume_registry_input() -> None:
    def unexpected_entries():
        raise AssertionError("disabled worker consumed registry entries")
        yield TagRegistryEntry(tag_id="unused", rider_id="unused")

    worker = build_scenario_worker(
        entries=unexpected_entries(),
        queue=InMemoryJobQueue(),
        clock=VirtualClock(NOW),
        assignment_policy=None,
        enabled=False,
    )

    assert worker is None


def test_trace_ordering_is_timestamp_sorted_and_stable() -> None:
    trace = order_trace_events(
        [
            (2.0, "later", {}),
            (1.0, "same-time-first", {"order": 1}),
            (1.0, "same-time-second", {"order": 2}),
        ]
    )

    assert tuple(item.kind for item in trace) == (
        "same-time-first",
        "same-time-second",
        "later",
    )
    assert tuple(item.sequence for item in trace) == (0, 1, 2)
