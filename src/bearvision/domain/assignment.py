"""Compatibility wrapper for the server-authoritative BearTag selection policy."""

from __future__ import annotations

from collections.abc import Iterable

from bearvision.contracts import (
    RiderAssignment,
    RiderAssignmentStatus,
    TagAssignmentEvidence,
    TagObservation,
)
from bearvision.ports import TagRegistry

from .tag_selection import TagSelectionStatus, select_bear_tag


def assign_rider(
    observations: Iterable[TagObservation],
    registry: TagRegistry,
    *,
    assigned_at_monotonic_s: float,
    clip_start_monotonic_s: float,
    clip_end_monotonic_s: float,
    minimum_observation_count: int = 2,
    minimum_motion_delta_mps2: float = 2.0,
    motion_full_scale_mps2: float = 12.0,
    minimum_rssi_dbm: int = -85,
    rssi_full_scale_dbm: int = -40,
    motion_weight: float = 0.7,
    rssi_weight: float = 0.3,
    minimum_score_margin: float = 0.12,
) -> RiderAssignment:
    """Preserve the former API while delegating all scoring to ``select_bear_tag``."""

    entries = {entry.tag_id: entry for entry in registry.entries() if entry.enabled}
    selection = select_bear_tag(
        observations,
        entries,
        clip_start_monotonic_s=clip_start_monotonic_s,
        clip_end_monotonic_s=clip_end_monotonic_s,
        minimum_observation_count=minimum_observation_count,
        minimum_motion_delta_mps2=minimum_motion_delta_mps2,
        motion_full_scale_mps2=motion_full_scale_mps2,
        minimum_rssi_dbm=minimum_rssi_dbm,
        rssi_full_scale_dbm=rssi_full_scale_dbm,
        motion_weight=motion_weight,
        rssi_weight=rssi_weight,
        minimum_score_margin=minimum_score_margin,
    )
    evidence = tuple(
        TagAssignmentEvidence(
            tag_id=item.bear_tag_id,
            rider_id=entries[item.bear_tag_id].rider_id,
            observation_count=item.observation_count,
            mean_motion_delta_mps2=item.mean_motion_delta_mps2,
            median_rssi_dbm=item.median_rssi_dbm,
            motion_score=item.motion_score,
            rssi_score=item.rssi_score,
            combined_score=item.combined_score,
            qualifies=item.qualifies,
        )
        for item in selection.evidence
    )
    if selection.status is TagSelectionStatus.UNASSIGNED:
        return RiderAssignment(
            status=RiderAssignmentStatus.UNASSIGNED,
            assigned_at_monotonic_s=assigned_at_monotonic_s,
            evidence=evidence,
            reason=selection.reason,
        )
    if selection.status is TagSelectionStatus.AMBIGUOUS:
        return RiderAssignment(
            status=RiderAssignmentStatus.AMBIGUOUS,
            assigned_at_monotonic_s=assigned_at_monotonic_s,
            candidate_tag_ids=selection.candidate_tag_ids,
            evidence=evidence,
            reason=selection.reason,
        )
    assert selection.selected_tag_id is not None
    winner = entries[selection.selected_tag_id]
    return RiderAssignment(
        status=RiderAssignmentStatus.ASSIGNED,
        assigned_at_monotonic_s=assigned_at_monotonic_s,
        rider_id=winner.rider_id,
        tag_id=winner.tag_id,
        candidate_tag_ids=selection.candidate_tag_ids,
        evidence=evidence,
        reason=selection.reason,
    )
