import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from bearvision.contracts import MediaAsset, TagObservation, Vector3
from bearvision.edge.job_package import build_edge_job
from bearvision.ports import CapturedMedia, ComponentUnavailable
from bearvision.server import (
    BearTagAssignment,
    FileSystemJobQueue,
    FileUserRegistry,
    ServerWorker,
    normalize_user_email,
)
from bearvision.simulation import VirtualClock


START = datetime(2026, 7, 1, 12, tzinfo=timezone.utc)


def media(job_id: str) -> CapturedMedia:
    content = f"video:{job_id}".encode()
    return CapturedMedia(
        asset=MediaAsset(
            asset_id=f"asset-{job_id}",
            filename="video.mp4",
            content_type="video/mp4",
            size_bytes=len(content),
            created_at_utc=START,
        ),
        content=content,
    )


def observation(tag_id: str, offset_s: float, *, rssi: int = -50) -> TagObservation:
    return TagObservation(
        tag_id=tag_id,
        observed_at_utc=START + timedelta(seconds=offset_s),
        observed_at_monotonic_s=offset_s,
        rssi_dbm=rssi,
        acceleration_mps2=Vector3(x=0, y=0, z=20),
    )


async def publish(queue: FileSystemJobQueue, job_id: str = "job-1") -> bool:
    clip = media(job_id)
    manifest, observations = build_edge_job(
        job_id=job_id,
        edge_device_id="edge-1",
        created_at=START + timedelta(seconds=5),
        capture_started_at=START,
        capture_ended_at=START + timedelta(seconds=5),
        clip_start_monotonic_s=0,
        video=clip,
        observations=(observation("BearTag-666", 1), observation("BearTag-666", 4)),
    )
    return await queue.publish(manifest, clip, observations)


def registry(path: Path, *, valid_to: datetime | None = None) -> FileUserRegistry:
    store = FileUserRegistry(path)
    user = store.create_user(" Bear.Eskildsen@GMAIL.com ", "Bear Eskildsen")
    store.create_bear_tag("BearTag-666")
    store.create_assignment(
        BearTagAssignment(
            id="assignment-1",
            userId=user.id,
            bearTagId="BearTag-666",
            validFrom=START - timedelta(days=1),
            validTo=valid_to or START + timedelta(days=1),
        )
    )
    return store


def test_complete_job_is_scored_and_moved_to_normalized_user(tmp_path: Path) -> None:
    queue = FileSystemJobQueue(tmp_path / "BearVision")
    store = registry(tmp_path / "registry.json")
    assert asyncio.run(publish(queue))

    result = asyncio.run(ServerWorker(queue, store, VirtualClock(START)).run_once())

    assert result is not None and result.status == "processed"
    assert result.selected_bear_tag_id == "BearTag-666"
    user_id = store.load().users[0].id
    assert result.selected_user_id == user_id
    result_path = (
        tmp_path
        / f"BearVision/processed/user_{user_id}/job-1/result.json"
    )
    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    assert persisted["assignmentId"] == "assignment-1"
    assert persisted["candidates"][0]["medianRssiDbm"] == -50
    assert queue.snapshot()["counts"]["processed"] == 1


def test_missing_assignment_and_boundary_are_unresolved(tmp_path: Path) -> None:
    for name, store, code in (
        ("missing", FileUserRegistry(tmp_path / "missing.json"), "NO_VALID_ASSIGNMENT"),
        (
            "boundary",
            registry(tmp_path / "boundary.json", valid_to=START + timedelta(seconds=2)),
            "ASSIGNMENT_BOUNDARY",
        ),
    ):
        queue = FileSystemJobQueue(tmp_path / name / "BearVision")
        if name == "missing":
            store.create_bear_tag("BearTag-666")
        asyncio.run(publish(queue, f"job-{name}"))
        result = asyncio.run(ServerWorker(queue, store, VirtualClock(START)).run_once())
        assert result is not None and result.status == "unresolved"
        assert result.error_code == code


def test_overlapping_assignment_is_rejected_without_partial_write(tmp_path: Path) -> None:
    store = registry(tmp_path / "registry.json")
    original = (tmp_path / "registry.json").read_bytes()

    with pytest.raises(ValueError, match="overlapping assignments"):
        store.create_assignment(
            BearTagAssignment(
                id="assignment-2",
                userId=store.load().users[0].id,
                bearTagId="BearTag-666",
                validFrom=START,
                validTo=START + timedelta(hours=1),
            )
        )

    assert (tmp_path / "registry.json").read_bytes() == original


def test_assignment_can_be_preflighted_without_writing_registry(tmp_path: Path) -> None:
    store = registry(tmp_path / "registry.json")
    store.create_bear_tag("BearTag-2")
    original = (tmp_path / "registry.json").read_bytes()

    proposed, _ = store.validate_assignment(
        BearTagAssignment(
            id="assignment-preview",
            userId=store.load().users[0].id,
            bearTagId="BearTag-2",
            validFrom=START,
            validTo=START + timedelta(hours=1),
        )
    )

    assert proposed.id == "assignment-preview"
    assert len(store.load().assignments) == 1
    assert (tmp_path / "registry.json").read_bytes() == original


def test_incomplete_and_duplicate_jobs_are_not_processed(tmp_path: Path) -> None:
    queue = FileSystemJobQueue(tmp_path / "BearVision")
    incomplete = tmp_path / "BearVision/input-queue/ready/incomplete"
    incomplete.mkdir()
    (incomplete / "manifest.json").write_text("{}", encoding="utf-8")
    assert asyncio.run(queue.acquire_next()) is None

    assert asyncio.run(publish(queue))
    assert not asyncio.run(publish(queue))


def test_processing_job_survives_queue_and_worker_restart(tmp_path: Path) -> None:
    queue_root = tmp_path / "BearVision"
    queue = FileSystemJobQueue(queue_root)
    store = registry(tmp_path / "registry.json")
    asyncio.run(publish(queue))
    assert asyncio.run(queue.acquire_next()) == "job-1"

    result = asyncio.run(
        ServerWorker(FileSystemJobQueue(queue_root), store, VirtualClock(START)).run_once()
    )

    assert result is not None and result.status == "processed"


def test_email_normalization_preserves_gmail_alias_semantics() -> None:
    assert normalize_user_email(" Bear.Name+Cable@GMAIL.com ") == "bear.name+cable@gmail.com"


@pytest.mark.parametrize("failure", ["job-id", "size", "checksum", "observation"])
def test_invalid_job_payload_is_failed_durably(tmp_path: Path, failure: str) -> None:
    queue = FileSystemJobQueue(tmp_path / failure / "BearVision")
    store = registry(tmp_path / failure / "registry.json")
    job_id = f"job-{failure}"
    assert asyncio.run(publish(queue, job_id))
    ready = tmp_path / failure / "BearVision/input-queue/ready" / job_id

    if failure == "job-id":
        manifest = json.loads((ready / "manifest.json").read_text(encoding="utf-8"))
        manifest["jobId"] = "different-job"
        (ready / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    elif failure == "size":
        (ready / "video.mp4").write_bytes(b"wrong-size")
    elif failure == "checksum":
        content = (ready / "video.mp4").read_bytes()
        (ready / "video.mp4").write_bytes(b"x" * len(content))
    else:
        (ready / "beartag-data.ndjson").write_text(
            '{"bearTagId":"BearTag-666","offsetMs":999999,'
            '"rssiDbm":-50,"accelerationMps2":{"x":0,"y":0,"z":20}}\n',
            encoding="utf-8",
        )

    result = asyncio.run(ServerWorker(queue, store, VirtualClock(START)).run_once())

    assert result is not None and result.status == "failed"
    assert result.error_code == "INVALID_JOB"
    assert queue.snapshot()["counts"]["failed"] == 1


def test_transient_queue_failure_keeps_claimed_job_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = FileSystemJobQueue(tmp_path / "BearVision")
    store = registry(tmp_path / "registry.json")
    assert asyncio.run(publish(queue))

    async def unavailable(*_args) -> bytes:
        raise ComponentUnavailable("temporary Box outage")

    monkeypatch.setattr(queue, "read", unavailable)
    with pytest.raises(ComponentUnavailable, match="temporary Box outage"):
        asyncio.run(ServerWorker(queue, store, VirtualClock(START)).run_once())

    assert queue.snapshot()["counts"]["processing"] == 1


def test_unexpected_queue_failure_becomes_technical_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = FileSystemJobQueue(tmp_path / "BearVision")
    store = registry(tmp_path / "registry.json")
    assert asyncio.run(publish(queue))

    async def broken(*_args) -> bytes:
        raise RuntimeError("unexpected adapter defect")

    monkeypatch.setattr(queue, "read", broken)
    result = asyncio.run(ServerWorker(queue, store, VirtualClock(START)).run_once())

    assert result is not None and result.status == "failed"
    assert result.error_code == "TECHNICAL_ERROR"
    assert queue.snapshot()["counts"]["failed"] == 1


def test_missing_package_file_is_failed_and_does_not_block_next_job(tmp_path: Path) -> None:
    queue = FileSystemJobQueue(tmp_path / "BearVision")
    store = registry(tmp_path / "registry.json")
    assert asyncio.run(publish(queue, "job-broken"))
    assert asyncio.run(publish(queue, "job-valid"))
    (tmp_path / "BearVision/input-queue/ready/job-broken/video.mp4").unlink()
    worker = ServerWorker(queue, store, VirtualClock(START))

    failed = asyncio.run(worker.run_once())
    processed = asyncio.run(worker.run_once())

    assert failed is not None and failed.error_code == "INVALID_JOB"
    assert processed is not None and processed.status == "processed"
    assert queue.snapshot()["counts"]["failed"] == 1
    assert queue.snapshot()["counts"]["processed"] == 1
