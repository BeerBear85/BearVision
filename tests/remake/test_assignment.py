from datetime import datetime, timezone

from bearvision.contracts import RiderAssignmentStatus, TagObservation, TagRegistryEntry, Vector3
from bearvision.domain import assign_rider
from bearvision.simulation import InMemoryTagRegistry


def observed(
    tag_id: str,
    at_s: float,
    rssi_dbm: int = -50,
    acceleration: Vector3 | None = None,
) -> TagObservation:
    return TagObservation(
        tag_id=tag_id,
        observed_at_utc=datetime(2026, 8, 12, tzinfo=timezone.utc),
        observed_at_monotonic_s=at_s,
        rssi_dbm=rssi_dbm,
        acceleration_mps2=acceleration or Vector3(x=0, y=0, z=19.0),
    )


def test_assignment_uses_ble_registry_only() -> None:
    registry = InMemoryTagRegistry((TagRegistryEntry(tag_id="tag-17", rider_id="rider-17"),))
    result = assign_rider((observed("tag-17", 3.8),), registry, assigned_at_monotonic_s=4)
    assert result.status is RiderAssignmentStatus.ASSIGNED
    assert result.rider_id == "rider-17"


def test_assignment_rejects_stale_weak_and_unknown_tags() -> None:
    registry = InMemoryTagRegistry((TagRegistryEntry(tag_id="tag-17", rider_id="rider-17"),))
    observations = (
        observed("tag-17", 1),
        observed("tag-17", 9, -100),
        observed("unknown", 9),
    )
    result = assign_rider(observations, registry, assigned_at_monotonic_s=10)
    assert result.status is RiderAssignmentStatus.UNASSIGNED


def test_assignment_preserves_multiple_ble_candidates() -> None:
    registry = InMemoryTagRegistry(
        (
            TagRegistryEntry(tag_id="tag-17", rider_id="rider-17"),
            TagRegistryEntry(tag_id="tag-22", rider_id="rider-22"),
        )
    )
    result = assign_rider(
        (observed("tag-22", 3), observed("tag-17", 3)),
        registry,
        assigned_at_monotonic_s=4,
    )
    assert result.status is RiderAssignmentStatus.AMBIGUOUS
    assert result.candidate_tag_ids == ("tag-17", "tag-22")
    assert result.rider_id is None


def test_stationary_nearby_rider_does_not_beat_active_rider() -> None:
    registry = InMemoryTagRegistry(
        (
            TagRegistryEntry(tag_id="active", rider_id="rider-active"),
            TagRegistryEntry(tag_id="nearby", rider_id="rider-nearby"),
        )
    )
    result = assign_rider(
        (
            observed("active", 4, -65, Vector3(x=5, y=2, z=19)),
            observed("nearby", 4, -40, Vector3(x=0, y=0, z=9.81)),
        ),
        registry,
        assigned_at_monotonic_s=4,
    )
    assert result.status is RiderAssignmentStatus.ASSIGNED
    assert result.rider_id == "rider-active"
    assert result.evidence[0].qualifies
    assert not result.evidence[1].qualifies


def test_rssi_disambiguates_two_tags_with_similar_jump_motion() -> None:
    registry = InMemoryTagRegistry(
        (
            TagRegistryEntry(tag_id="strong", rider_id="rider-strong"),
            TagRegistryEntry(tag_id="weak", rider_id="rider-weak"),
        )
    )
    result = assign_rider(
        (observed("strong", 4, -45), observed("weak", 4, -75)),
        registry,
        assigned_at_monotonic_s=4,
    )
    assert result.status is RiderAssignmentStatus.ASSIGNED
    assert result.rider_id == "rider-strong"
    assert len(result.candidate_tag_ids) == 2
