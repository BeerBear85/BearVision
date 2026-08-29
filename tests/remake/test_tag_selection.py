from datetime import datetime, timezone

from bearvision.contracts import TagObservation, Vector3
from bearvision.domain import TagSelectionStatus, select_bear_tag


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


def select(
    observations: tuple[TagObservation, ...],
    known_tag_ids: tuple[str, ...],
    *,
    start_s: float,
    end_s: float,
):
    return select_bear_tag(
        observations,
        known_tag_ids,
        clip_start_monotonic_s=start_s,
        clip_end_monotonic_s=end_s,
    )


def test_selection_uses_server_registry_tag_ids() -> None:
    result = select(
        (observed("tag-17", 4), observed("tag-17", 4.1)),
        ("tag-17",),
        start_s=4,
        end_s=9,
    )

    assert result.status is TagSelectionStatus.SELECTED
    assert result.selected_tag_id == "tag-17"


def test_selection_rejects_stale_weak_and_unknown_tags() -> None:
    result = select(
        (
            observed("tag-17", 1),
            observed("tag-17", 9, -100),
            observed("unknown", 9),
        ),
        ("tag-17",),
        start_s=8,
        end_s=10,
    )

    assert result.status is TagSelectionStatus.UNASSIGNED


def test_selection_preserves_multiple_candidates() -> None:
    result = select(
        (
            observed("tag-22", 3),
            observed("tag-22", 3.1),
            observed("tag-17", 3),
            observed("tag-17", 3.1),
        ),
        ("tag-17", "tag-22"),
        start_s=3,
        end_s=4,
    )

    assert result.status is TagSelectionStatus.AMBIGUOUS
    assert result.candidate_tag_ids == ("tag-17", "tag-22")


def test_stationary_nearby_tag_does_not_beat_active_tag() -> None:
    result = select(
        (
            observed("active", 4, -65, Vector3(x=5, y=2, z=19)),
            observed("active", 4.1, -66, Vector3(x=4, y=2, z=18)),
            observed("nearby", 4, -40, Vector3(x=0, y=0, z=9.81)),
            observed("nearby", 4.1, -41, Vector3(x=0, y=0, z=9.8)),
        ),
        ("active", "nearby"),
        start_s=4,
        end_s=5,
    )

    assert result.status is TagSelectionStatus.SELECTED
    assert result.selected_tag_id == "active"
    assert result.evidence[0].qualifies
    assert not result.evidence[1].qualifies


def test_rssi_disambiguates_similarly_active_tags() -> None:
    result = select(
        (
            observed("strong", 4, -45),
            observed("strong", 4.1, -46),
            observed("weak", 4, -75),
            observed("weak", 4.1, -76),
        ),
        ("strong", "weak"),
        start_s=4,
        end_s=5,
    )

    assert result.status is TagSelectionStatus.SELECTED
    assert result.selected_tag_id == "strong"
    assert len(result.candidate_tag_ids) == 2


def test_motion_is_meaned_across_every_sample_in_clip() -> None:
    result = select(
        (
            observed("tag-17", 1, acceleration=Vector3(x=0, y=0, z=19)),
            observed("tag-17", 2, acceleration=Vector3(x=0, y=0, z=9.80665)),
        ),
        ("tag-17",),
        start_s=1,
        end_s=3,
    )

    assert 4.5 < result.evidence[0].mean_motion_delta_mps2 < 4.7
