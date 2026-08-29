import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from bearvision.adapters import BoxJobQueue
from bearvision.contracts import JobResultManifest, MediaAsset, TagObservation, Vector3
from bearvision.edge.job_package import build_edge_job
from bearvision.ports import CapturedMedia


USER_ID = UUID("b10e3918-490c-4a3f-859a-e67c12b66680")


class MemoryBoxFolders:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def upload_file(self, local_path, remote_path, overwrite=False):
        if remote_path in self.files and not overwrite:
            raise FileExistsError(remote_path)
        self.files[remote_path] = Path(local_path).read_bytes()

    def download_file(self, remote_path, local_path):
        if remote_path not in self.files:
            raise FileNotFoundError(remote_path)
        Path(local_path).write_bytes(self.files[remote_path])

    def folder_exists(self, path):
        prefix = path.strip("/") + "/"
        return any(item.startswith(prefix) for item in self.files)

    def file_exists(self, path):
        return path.strip("/") in self.files

    def list_folders(self, path=""):
        prefix = path.strip("/")
        prefix = f"{prefix}/" if prefix else ""
        return sorted(
            {
                item.removeprefix(prefix).split("/", 1)[0]
                for item in self.files
                if item.startswith(prefix) and "/" in item.removeprefix(prefix)
            }
        )

    def move_folder(self, source, destination):
        source_prefix = source.strip("/") + "/"
        destination_prefix = destination.strip("/") + "/"
        moving = {key: value for key, value in self.files.items() if key.startswith(source_prefix)}
        if not moving:
            raise FileNotFoundError(source)
        for key in moving:
            del self.files[key]
        for key, value in moving.items():
            self.files[destination_prefix + key.removeprefix(source_prefix)] = value

    def delete_file(self, path):
        del self.files[path.strip("/")]

    def delete_folder(self, path):
        prefix = path.strip("/") + "/"
        for item in [item for item in self.files if item.startswith(prefix)]:
            del self.files[item]


def test_box_queue_commits_with_ready_then_claims_complete_folder(tmp_path: Path) -> None:
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    video = CapturedMedia(
        asset=MediaAsset(
            asset_id="asset-1",
            filename="video.mp4",
            content_type="video/mp4",
            size_bytes=5,
            created_at_utc=now,
        ),
        content=b"video",
    )
    source_observation = TagObservation(
        tag_id="BearTag-1",
        observed_at_utc=now + timedelta(seconds=1),
        observed_at_monotonic_s=1,
        rssi_dbm=-50,
        acceleration_mps2=Vector3(x=1, y=2, z=3),
    )
    manifest, observations = build_edge_job(
        job_id="job-1",
        edge_device_id="edge-1",
        created_at=now + timedelta(seconds=5),
        capture_started_at=now,
        capture_ended_at=now + timedelta(seconds=5),
        clip_start_monotonic_s=0,
        video=video,
        observations=(source_observation,),
    )
    handler = MemoryBoxFolders()
    queue = BoxJobQueue(handler, tmp_path)

    # A previous transient upload can be resumed and committed.
    handler.files["input-queue/uploading/job-1/video.mp4"] = b"partial"

    assert asyncio.run(queue.publish(manifest, video, observations))
    assert "input-queue/ready/job-1/READY" in handler.files
    assert not any(key.startswith("input-queue/uploading") for key in handler.files)
    assert asyncio.run(queue.acquire_next()) == "job-1"
    assert asyncio.run(queue.read("job-1", "video.mp4")) == b"video"
    assert not asyncio.run(queue.publish(manifest, video, observations))
    assert queue.admin_list_jobs() == [{"jobId": "job-1", "status": "processing"}]
    assert asyncio.run(queue.admin_read("job-1", "manifest.json"))
    downloaded = tmp_path / "downloaded.mp4"
    asyncio.run(queue.admin_download("job-1", "video.mp4", downloaded))
    assert downloaded.read_bytes() == b"video"

    processed = JobResultManifest(
        jobId="job-1",
        status="processed",
        processedAt=now + timedelta(seconds=6),
        algorithmVersion="test-v1",
        selectedBearTagId="BearTag-1",
        selectedUserId=USER_ID,
        assignmentId="assignment-1",
        reason="selected test rider",
    )
    asyncio.run(queue.finish("job-1", processed, USER_ID))

    assert queue.admin_list_jobs() == [
        {
            "jobId": "job-1",
            "status": "processed",
            "userId": str(USER_ID),
        }
    ]
    snapshot = queue.snapshot()
    assert snapshot["counts"]["processed"] == 1
    assert snapshot["jobs"][0]["selectedUserId"] == str(USER_ID)

    second_manifest = manifest.model_copy(update={"job_id": "job-2"})
    assert asyncio.run(queue.publish(second_manifest, video, observations))
    assert asyncio.run(queue.acquire_next()) == "job-2"
    unresolved = JobResultManifest(
        jobId="job-2",
        status="unresolved",
        processedAt=now + timedelta(seconds=7),
        algorithmVersion="test-v1",
        reason="no unique rider",
        errorCode="AMBIGUOUS_BEARTAG",
    )
    asyncio.run(queue.finish("job-2", unresolved))
    assert queue.snapshot()["counts"]["unresolved"] == 1
    assert asyncio.run(queue.requeue("job-2"))
    assert not asyncio.run(queue.requeue("missing-job"))
    assert asyncio.run(queue.acquire_next()) == "job-2"


def test_box_queue_preserves_missing_package_file_as_permanent_error(tmp_path: Path) -> None:
    handler = MemoryBoxFolders()
    handler.files["processing/job-broken/manifest.json"] = b"{}"
    queue = BoxJobQueue(handler, tmp_path)

    with pytest.raises(FileNotFoundError):
        asyncio.run(queue.read("job-broken", "video.mp4"))
