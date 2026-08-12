"""BLE-only rider assignment policy."""

from __future__ import annotations

from collections.abc import Iterable

from bearvision.contracts import (
    RiderAssignment,
    RiderAssignmentStatus,
    TagObservation,
)
from bearvision.ports import TagRegistry


def assign_rider(
    observations: Iterable[TagObservation],
    registry: TagRegistry,
    *,
    assigned_at_monotonic_s: float,
    maximum_age_s: float = 5.0,
    minimum_rssi_dbm: int = -80,
) -> RiderAssignment:
    """Assign exactly one qualifying registered BLE tag, or preserve uncertainty."""

    if maximum_age_s < 0:
        raise ValueError("maximum_age_s must not be negative")

    qualifying: dict[str, TagObservation] = {}
    for observation in observations:
        age_s = assigned_at_monotonic_s - observation.observed_at_monotonic_s
        entry = registry.resolve(observation.tag_id)
        if (
            entry is not None
            and entry.enabled
            and 0 <= age_s <= maximum_age_s
            and observation.rssi_dbm >= minimum_rssi_dbm
        ):
            previous = qualifying.get(observation.tag_id)
            if previous is None or observation.observed_at_monotonic_s > previous.observed_at_monotonic_s:
                qualifying[observation.tag_id] = observation

    candidate_ids = tuple(sorted(qualifying))
    if not candidate_ids:
        return RiderAssignment(
            status=RiderAssignmentStatus.UNASSIGNED,
            assigned_at_monotonic_s=assigned_at_monotonic_s,
            reason="no recent registered BLE tag qualifies",
        )
    if len(candidate_ids) > 1:
        return RiderAssignment(
            status=RiderAssignmentStatus.AMBIGUOUS,
            assigned_at_monotonic_s=assigned_at_monotonic_s,
            candidate_tag_ids=candidate_ids,
            reason="multiple recent registered BLE tags qualify",
        )

    tag_id = candidate_ids[0]
    entry = registry.resolve(tag_id)
    assert entry is not None
    return RiderAssignment(
        status=RiderAssignmentStatus.ASSIGNED,
        assigned_at_monotonic_s=assigned_at_monotonic_s,
        rider_id=entry.rider_id,
        tag_id=tag_id,
        candidate_tag_ids=(tag_id,),
        reason="one recent registered BLE tag qualifies",
    )
