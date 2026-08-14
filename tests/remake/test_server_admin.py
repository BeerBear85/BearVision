import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bearvision.contracts import MediaAsset, TagObservation, Vector3
from bearvision.edge.job_package import build_edge_job
from bearvision.ports import CapturedMedia
from bearvision.server.admin import AdminCatalog, AdminMediaService, UserVideoCatalog
from bearvision.server.queue import FileSystemJobQueue
from bearvision.server.registry import BearTagAssignment, FileUserRegistry
from bearvision.server.worker import ServerWorker
from bearvision.simulation import VirtualClock


START = datetime(2026, 8, 13, 8, tzinfo=timezone.utc)


async def processed_fixture(
    tmp_path: Path,
) -> tuple[FileSystemJobQueue, FileUserRegistry, bytes]:
    queue = FileSystemJobQueue(tmp_path / "queue")
    registry = FileUserRegistry(tmp_path / "registry.json")
    user = registry.create_user(" Bear@Example.com ", "Bear Rider")
    registry.create_bear_tag("BearTag-1")
    registry.create_assignment(
        BearTagAssignment(
            id="assignment-1",
            userId=user.id,
            bearTagId="BearTag-1",
            validFrom=START - timedelta(hours=1),
            validTo=START + timedelta(hours=1),
        )
    )
    content = b"test-video-content"
    media = CapturedMedia(
        asset=MediaAsset(
            asset_id="asset-1",
            filename="clip.mp4",
            content_type="video/mp4",
            size_bytes=len(content),
            created_at_utc=START,
        ),
        content=content,
    )
    observations = tuple(
        TagObservation(
            tag_id="BearTag-1",
            observed_at_utc=START + timedelta(seconds=offset),
            observed_at_monotonic_s=float(offset),
            rssi_dbm=-48,
            acceleration_mps2=Vector3(x=0, y=0, z=20),
        )
        for offset in (1, 2)
    )
    manifest, packaged = build_edge_job(
        job_id="job-20260813-001",
        edge_device_id="edge-1",
        created_at=START + timedelta(seconds=4),
        capture_started_at=START,
        capture_ended_at=START + timedelta(seconds=4),
        clip_start_monotonic_s=0,
        video=media,
        observations=observations,
    )
    assert await queue.publish(manifest, media, packaged)
    result = await ServerWorker(queue, registry, VirtualClock(START)).run_once()
    assert result is not None and result.status == "processed"
    return queue, registry, content


def test_admin_catalog_pages_and_enriches_jobs_and_users(tmp_path: Path) -> None:
    queue, registry, _ = asyncio.run(processed_fixture(tmp_path))
    catalog = AdminCatalog(queue, registry)

    jobs = asyncio.run(
        catalog.list_jobs(status="processed", query="BearTag-1", page_size=10)
    )
    assert jobs["total"] == 1
    assert jobs["items"][0]["displayName"] == "Bear Rider"
    assert jobs["items"][0]["durationSeconds"] == 4
    assert jobs["items"][0]["selectedBearTagId"] == "BearTag-1"

    users = asyncio.run(catalog.list_users(query="bear", page_size=10))
    assert users["items"][0]["email"] == "bear@example.com"
    assert users["items"][0]["processedVideoCount"] == 1
    assert users["items"][0]["assignments"][0]["bearTagId"] == "BearTag-1"


class StubThumbnailMediaService(AdminMediaService):
    def _create_thumbnail(self, video: Path, destination: Path) -> None:
        assert video.read_bytes() == b"test-video-content"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"jpeg")


def test_python_materializes_verified_video_and_thumbnail(tmp_path: Path) -> None:
    queue, _, content = asyncio.run(processed_fixture(tmp_path))
    service = StubThumbnailMediaService(queue, tmp_path / "cache")

    video = asyncio.run(service.materialize("job-20260813-001", "video"))
    thumbnail = asyncio.run(service.materialize("job-20260813-001", "thumbnail"))

    assert Path(video["path"]).read_bytes() == content
    assert video["contentType"] == "video/mp4"
    assert Path(thumbnail["path"]).read_bytes() == b"jpeg"
    assert thumbnail["contentType"] == "image/jpeg"


def test_user_catalog_exposes_only_public_video_fields(tmp_path: Path) -> None:
    queue, registry, _ = asyncio.run(processed_fixture(tmp_path))

    result = asyncio.run(
        UserVideoCatalog(queue, registry).list_videos(" Bear@Example.com ")
    )

    assert result["user"] == {
        "id": str(registry.load().users[0].id),
        "email": "bear@example.com",
        "displayName": "Bear Rider",
    }
    assert result["total"] == 1
    assert result["items"][0]["jobId"] == "job-20260813-001"
    assert "selectedUserEmail" not in result["items"][0]
    assert "selectedBearTagId" not in result["items"][0]


def test_user_media_rejects_a_job_owned_by_someone_else(tmp_path: Path) -> None:
    queue, registry, _ = asyncio.run(processed_fixture(tmp_path))
    service = StubThumbnailMediaService(
        queue, tmp_path / "cache", registry=registry
    )

    with pytest.raises(FileNotFoundError, match="video not found for user"):
        asyncio.run(
            service.materialize_for_user(
                "someone-else@example.com", "job-20260813-001", "video"
            )
        )


def test_failed_job_with_broken_metadata_remains_visible(tmp_path: Path) -> None:
    queue = FileSystemJobQueue(tmp_path / "queue")
    failed = tmp_path / "queue/failed/broken-job"
    failed.mkdir()
    (failed / "manifest.json").write_text("{}", encoding="utf-8")
    registry = FileUserRegistry(tmp_path / "registry.json")

    jobs = asyncio.run(AdminCatalog(queue, registry).list_jobs(status="failed"))

    assert jobs["total"] == 1
    assert jobs["items"][0]["jobId"] == "broken-job"
    assert len(jobs["items"][0]["metadataErrors"]) == 2


def test_packaged_ffmpeg_generates_real_thumbnail(tmp_path: Path) -> None:
    pytest.importorskip("ffmpeg_binaries")
    source = Path("tests/data/preview_low.mp4")
    content = source.read_bytes()
    queue = FileSystemJobQueue(tmp_path / "queue")
    media = CapturedMedia(
        asset=MediaAsset(
            asset_id="preview-asset",
            filename="preview.mp4",
            content_type="video/mp4",
            size_bytes=len(content),
            created_at_utc=START,
        ),
        content=content,
    )
    manifest, observations = build_edge_job(
        job_id="preview-job",
        edge_device_id="edge-1",
        created_at=START + timedelta(seconds=4),
        capture_started_at=START,
        capture_ended_at=START + timedelta(seconds=4),
        clip_start_monotonic_s=0,
        video=media,
        observations=(),
    )
    assert asyncio.run(queue.publish(manifest, media, observations))

    result = asyncio.run(
        AdminMediaService(queue, tmp_path / "cache").materialize(
            "preview-job", "thumbnail"
        )
    )

    thumbnail = Path(result["path"]).read_bytes()
    assert thumbnail.startswith(b"\xff\xd8")
    assert len(thumbnail) > 1000
