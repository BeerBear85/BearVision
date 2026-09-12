import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bearvision.config import AssignmentConfig
from bearvision.contracts import JobResultManifest, MediaAsset, TagObservation, Vector3
from bearvision.edge.job_package import build_edge_job
from bearvision.ports import CapturedMedia
from bearvision.server.corrections import AssignmentCorrectionService
from bearvision.server.queue import FileSystemJobQueue
from bearvision.server.registry import BearTagAssignment, FileUserRegistry
from bearvision.server.worker import ServerWorker
from bearvision.simulation import VirtualClock


START = datetime(2026, 9, 11, 10, tzinfo=timezone.utc)


def job_package(job_id: str):
    content = ("video-" + job_id).encode()
    media = CapturedMedia(
        asset=MediaAsset(
            asset_id="asset-" + job_id,
            filename="clip.mp4",
            content_type="video/mp4",
            size_bytes=len(content),
            created_at_utc=START,
        ),
        content=content,
    )
    observations = tuple(
        TagObservation(
            tag_id="tag-17",
            observed_at_utc=START + timedelta(seconds=offset),
            observed_at_monotonic_s=float(offset),
            rssi_dbm=-45,
            acceleration_mps2=Vector3(x=0, y=0, z=20),
        )
        for offset in (1, 2)
    )
    manifest, packaged = build_edge_job(
        job_id=job_id,
        edge_device_id="edge-1",
        created_at=START + timedelta(seconds=5),
        capture_started_at=START,
        capture_ended_at=START + timedelta(seconds=4),
        clip_start_monotonic_s=0,
        video=media,
        observations=observations,
    )
    return manifest, media, packaged, content


def correction_fixture(tmp_path: Path):
    queue = FileSystemJobQueue(tmp_path / "queue")
    registry = FileUserRegistry(tmp_path / "registry.json")
    alpha = registry.create_user("alpha@example.com", "Alpha")
    bravo = registry.create_user("bravo@example.com", "Bravo")
    registry.create_bear_tag("tag-17")
    original = registry.create_assignment(
        BearTagAssignment(
            id="history-1",
            userId=alpha.id,
            bearTagId="tag-17",
            validFrom=START - timedelta(hours=1),
            validTo=START + timedelta(hours=1),
        )
    )
    manifest, media, observations, content = job_package("processed-job")
    assert asyncio.run(queue.publish(manifest, media, observations))
    result = asyncio.run(ServerWorker(queue, registry, VirtualClock(START)).run_once())
    assert result is not None and result.status == "processed"

    manifest, media, observations, unresolved_content = job_package("unresolved-job")
    assert asyncio.run(queue.publish(manifest, media, observations))
    assert asyncio.run(queue.acquire_next()) == "unresolved-job"
    asyncio.run(
        queue.finish(
            "unresolved-job",
            JobResultManifest(
                jobId="unresolved-job",
                status="unresolved",
                processedAt=START,
                algorithmVersion="test",
                selectedBearTagId="tag-17",
                reason="needs operator",
                errorCode="NO_VALID_ASSIGNMENT",
            ),
        )
    )
    return queue, registry, alpha, bravo, original, content, unresolved_content


def moved(original, user_id, assignment_id="history-1"):
    return (
        BearTagAssignment(
            id=assignment_id,
            userId=user_id,
            bearTagId=original.bear_tag_id,
            validFrom=original.valid_from,
            validTo=original.valid_to,
        ),
    )


def test_manual_reassignment_handles_processed_and_unresolved_without_losing_media(
    tmp_path: Path,
) -> None:
    queue, registry, alpha, bravo, _, processed_bytes, unresolved_bytes = correction_fixture(
        tmp_path
    )
    service = AssignmentCorrectionService(queue, registry, AssignmentConfig())

    processed = asyncio.run(service.manual_reassign("processed-job", bravo.id, "Wrong rider"))
    unresolved = asyncio.run(service.manual_reassign("unresolved-job", alpha.id, "Known rider"))

    snapshot = queue.snapshot()
    assert snapshot["counts"]["processed"] == 2
    assert processed["replacedUserId"] == str(alpha.id)
    assert unresolved["replacedUserId"] is None
    for job_id, expected in (
        ("processed-job", processed_bytes),
        ("unresolved-job", unresolved_bytes),
    ):
        result = JobResultManifest.model_validate_json(
            asyncio.run(queue.admin_read(job_id, "result.json"))
        )
        assert result.assignment_source == "manual"
        assert result.manual_reason
        assert result.manually_assigned_at
        assert asyncio.run(queue.admin_read(job_id, "clip.mp4")) == expected


def test_user_name_and_email_edit_preserves_uuid_and_audits_previous_values(tmp_path: Path) -> None:
    registry = FileUserRegistry(tmp_path / "registry.json")
    user = registry.create_user("before@example.com", "Before")

    updated = registry.update_user(
        user.id,
        email="after@example.com",
        display_name="After",
        reason="Correct identity",
        recorded_at=START,
    )

    assert updated.id == user.id
    assert (updated.email, updated.display_name) == ("after@example.com", "After")
    audit = registry.load().audit_trail[-1]
    assert audit.before["email"] == "before@example.com"
    assert audit.after["displayName"] == "After"


def test_assignment_history_can_move_or_split_and_rejects_overlap_and_gaps(tmp_path: Path) -> None:
    _, registry, alpha, bravo, original, *_ = correction_fixture(tmp_path)
    midpoint = START
    split = (
        BearTagAssignment(
            id="history-1:1",
            userId=alpha.id,
            bearTagId="tag-17",
            validFrom=original.valid_from,
            validTo=midpoint,
        ),
        BearTagAssignment(
            id="history-1:2",
            userId=bravo.id,
            bearTagId="tag-17",
            validFrom=midpoint,
            validTo=original.valid_to,
        ),
    )
    registry.replace_assignment("history-1", split, reason="Handover corrected", recorded_at=START)
    assert [item.user_id for item in registry.load().assignments] == [alpha.id, bravo.id]
    assert registry.load().audit_trail[-1].action == "assignment-history-updated"

    with pytest.raises(ValueError, match="overlap"):
        registry.preview_assignment_replacement(
            "history-1:1",
            (
                BearTagAssignment(
                    id="overlap-1",
                    userId=alpha.id,
                    bearTagId="tag-17",
                    validFrom=original.valid_from,
                    validTo=midpoint,
                ),
                BearTagAssignment(
                    id="overlap-2",
                    userId=bravo.id,
                    bearTagId="tag-17",
                    validFrom=midpoint - timedelta(minutes=1),
                    validTo=midpoint,
                ),
            ),
        )
    with pytest.raises(ValueError, match="gap"):
        registry.preview_assignment_replacement(
            "history-1:1",
            (
                BearTagAssignment(
                    id="gap-1",
                    userId=alpha.id,
                    bearTagId="tag-17",
                    validFrom=original.valid_from,
                    validTo=midpoint - timedelta(minutes=2),
                ),
                BearTagAssignment(
                    id="gap-2",
                    userId=bravo.id,
                    bearTagId="tag-17",
                    validFrom=midpoint - timedelta(minutes=1),
                    validTo=midpoint,
                ),
            ),
        )


def test_impact_preview_and_batch_recalculation_show_before_after_and_preserve_media(
    tmp_path: Path,
) -> None:
    queue, registry, alpha, bravo, original, content, _ = correction_fixture(tmp_path)
    service = AssignmentCorrectionService(queue, registry, AssignmentConfig())
    replacements = moved(original, bravo.id)

    preview = asyncio.run(
        service.preview_history_change("history-1", replacements, override_manual_assignments=False)
    )
    item = next(item for item in preview["items"] if item["jobId"] == "processed-job")
    assert item["current"] == {**item["current"], "userId": str(alpha.id), "displayName": "Alpha"}
    assert item["expected"]["userId"] == str(bravo.id)
    assert preview["counts"]["changed"] >= 1

    result = asyncio.run(
        service.apply_history_change(
            "history-1", replacements, reason="Correct owner", override_manual_assignments=False
        )
    )
    assert result["applied"] is True
    assert result["counts"]["changed"] >= 1
    assert asyncio.run(queue.admin_read("processed-job", "clip.mp4")) == content
    assert next(item for item in queue.snapshot()["jobs"] if item["jobId"] == "processed-job")[
        "userId"
    ] == str(bravo.id)


def test_manual_assignments_are_protected_unless_operator_explicitly_overrides(
    tmp_path: Path,
) -> None:
    queue, registry, alpha, bravo, original, *_ = correction_fixture(tmp_path)
    service = AssignmentCorrectionService(queue, registry, AssignmentConfig())
    asyncio.run(service.manual_reassign("processed-job", bravo.id, "Operator knows rider"))
    replacements = moved(original, alpha.id)

    protected = asyncio.run(
        service.preview_history_change("history-1", replacements, override_manual_assignments=False)
    )
    protected_item = next(item for item in protected["items"] if item["jobId"] == "processed-job")
    assert protected_item["manualProtected"] is True
    assert protected_item["changed"] is False

    overridden = asyncio.run(
        service.preview_history_change("history-1", replacements, override_manual_assignments=True)
    )
    overridden_item = next(item for item in overridden["items"] if item["jobId"] == "processed-job")
    assert overridden_item["manualProtected"] is False
    assert overridden_item["changed"] is True
    assert overridden_item["expected"]["assignmentSource"] == "automatic"
