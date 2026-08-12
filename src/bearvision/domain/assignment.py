"""Whole-clip BearTag acceleration-plus-RSSI rider assignment policy."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from statistics import fmean, median

from bearvision.contracts import (
    RiderAssignment,
    RiderAssignmentStatus,
    TagAssignmentEvidence,
    TagObservation,
)
from bearvision.ports import TagRegistry


STANDARD_GRAVITY_MPS2 = 9.80665


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


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
    """Fuse each registered BearTag's evidence over the complete recorded clip.

    Acceleration is averaged as orientation-independent activity magnitude
    ``abs(norm(a) - g)``. Averaging signed axes would let movement cancel itself.
    RSSI uses a median over the identical interval to suppress radio outliers.
    """

    if clip_start_monotonic_s < 0 or clip_end_monotonic_s < clip_start_monotonic_s:
        raise ValueError("clip interval is invalid")
    if minimum_observation_count < 1:
        raise ValueError("minimum_observation_count must be positive")
    if motion_full_scale_mps2 <= minimum_motion_delta_mps2:
        raise ValueError("motion_full_scale_mps2 must exceed the motion threshold")
    if rssi_full_scale_dbm <= minimum_rssi_dbm:
        raise ValueError("rssi_full_scale_dbm must exceed minimum_rssi_dbm")
    if not math.isclose(motion_weight + rssi_weight, 1.0, abs_tol=1e-9):
        raise ValueError("motion_weight and rssi_weight must sum to 1.0")

    grouped: dict[str, list[TagObservation]] = defaultdict(list)
    for observation in observations:
        if not clip_start_monotonic_s <= observation.observed_at_monotonic_s <= clip_end_monotonic_s:
            continue
        entry = registry.resolve(observation.tag_id)
        if entry is not None and entry.enabled:
            grouped[observation.tag_id].append(observation)

    evidence: list[TagAssignmentEvidence] = []
    for tag_id, samples in grouped.items():
        entry = registry.resolve(tag_id)
        assert entry is not None
        motion_deltas = [
            abs(
                math.sqrt(
                    sample.acceleration_mps2.x**2
                    + sample.acceleration_mps2.y**2
                    + sample.acceleration_mps2.z**2
                )
                - STANDARD_GRAVITY_MPS2
            )
            for sample in samples
        ]
        mean_motion = fmean(motion_deltas)
        median_rssi = float(median(sample.rssi_dbm for sample in samples))
        motion_score = _clamp(
            (mean_motion - minimum_motion_delta_mps2)
            / (motion_full_scale_mps2 - minimum_motion_delta_mps2)
        )
        rssi_score = _clamp(
            (median_rssi - minimum_rssi_dbm)
            / (rssi_full_scale_dbm - minimum_rssi_dbm)
        )
        qualifies = (
            len(samples) >= minimum_observation_count
            and mean_motion >= minimum_motion_delta_mps2
            and median_rssi >= minimum_rssi_dbm
        )
        evidence.append(
            TagAssignmentEvidence(
                tag_id=tag_id,
                rider_id=entry.rider_id,
                observation_count=len(samples),
                mean_motion_delta_mps2=mean_motion,
                median_rssi_dbm=median_rssi,
                motion_score=motion_score,
                rssi_score=rssi_score,
                combined_score=motion_weight * motion_score + rssi_weight * rssi_score,
                qualifies=qualifies,
            )
        )

    evidence.sort(key=lambda item: (-item.combined_score, item.tag_id))
    qualified = [item for item in evidence if item.qualifies]
    candidate_ids = tuple(sorted(item.tag_id for item in qualified))
    evidence_tuple = tuple(evidence)
    if not qualified:
        return RiderAssignment(
            status=RiderAssignmentStatus.UNASSIGNED,
            assigned_at_monotonic_s=assigned_at_monotonic_s,
            evidence=evidence_tuple,
            reason="no registered BearTag passes whole-clip observation, motion and RSSI gates",
        )

    winner = qualified[0]
    if len(qualified) > 1 and winner.combined_score - qualified[1].combined_score < minimum_score_margin:
        return RiderAssignment(
            status=RiderAssignmentStatus.AMBIGUOUS,
            assigned_at_monotonic_s=assigned_at_monotonic_s,
            candidate_tag_ids=candidate_ids,
            evidence=evidence_tuple,
            reason=(
                "multiple active BearTags have whole-clip acceleration/RSSI scores "
                f"within the {minimum_score_margin:.3f} decision margin"
            ),
        )

    return RiderAssignment(
        status=RiderAssignmentStatus.ASSIGNED,
        assigned_at_monotonic_s=assigned_at_monotonic_s,
        rider_id=winner.rider_id,
        tag_id=winner.tag_id,
        candidate_tag_ids=candidate_ids,
        evidence=evidence_tuple,
        reason="BearTag has the strongest whole-clip mean-motion and RSSI evidence",
    )
