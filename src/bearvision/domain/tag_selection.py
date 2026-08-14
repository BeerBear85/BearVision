"""Authoritative whole-clip BearTag score and winner selection."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from statistics import fmean, median

from bearvision.contracts import CandidateScore, TagObservation


ALGORITHM_VERSION = "beartag-fusion-1.0"
STANDARD_GRAVITY_MPS2 = 9.80665


class TagSelectionStatus(StrEnum):
    SELECTED = "selected"
    UNASSIGNED = "unassigned"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class TagSelection:
    status: TagSelectionStatus
    selected_tag_id: str | None
    candidate_tag_ids: tuple[str, ...]
    evidence: tuple[CandidateScore, ...]
    reason: str


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def select_bear_tag(
    observations: Iterable[TagObservation],
    known_tag_ids: Iterable[str],
    *,
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
) -> TagSelection:
    """Apply the existing motion/RSSI algorithm without resolving a user."""

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

    known = set(known_tag_ids)
    grouped: dict[str, list[TagObservation]] = defaultdict(list)
    for observation in observations:
        if (
            observation.tag_id in known
            and clip_start_monotonic_s
            <= observation.observed_at_monotonic_s
            <= clip_end_monotonic_s
        ):
            grouped[observation.tag_id].append(observation)

    evidence: list[CandidateScore] = []
    for tag_id, samples in grouped.items():
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
            (median_rssi - minimum_rssi_dbm) / (rssi_full_scale_dbm - minimum_rssi_dbm)
        )
        evidence.append(
            CandidateScore(
                bearTagId=tag_id,
                observationCount=len(samples),
                meanMotionDeltaMps2=mean_motion,
                medianRssiDbm=median_rssi,
                motionScore=motion_score,
                rssiScore=rssi_score,
                combinedScore=motion_weight * motion_score + rssi_weight * rssi_score,
                qualifies=(
                    len(samples) >= minimum_observation_count
                    and mean_motion >= minimum_motion_delta_mps2
                    and median_rssi >= minimum_rssi_dbm
                ),
            )
        )

    evidence.sort(key=lambda item: (-item.combined_score, item.bear_tag_id))
    qualified = [item for item in evidence if item.qualifies]
    candidate_ids = tuple(sorted(item.bear_tag_id for item in qualified))
    if not qualified:
        return TagSelection(
            TagSelectionStatus.UNASSIGNED,
            None,
            (),
            tuple(evidence),
            "no registered BearTag passes whole-clip observation, motion and RSSI gates",
        )
    winner = qualified[0]
    if (
        len(qualified) > 1
        and winner.combined_score - qualified[1].combined_score < minimum_score_margin
    ):
        return TagSelection(
            TagSelectionStatus.AMBIGUOUS,
            None,
            candidate_ids,
            tuple(evidence),
            "multiple active BearTags have whole-clip acceleration/RSSI scores "
            f"within the {minimum_score_margin:.3f} decision margin",
        )
    return TagSelection(
        TagSelectionStatus.SELECTED,
        winner.bear_tag_id,
        candidate_ids,
        tuple(evidence),
        "BearTag has the strongest whole-clip mean-motion and RSSI evidence",
    )
