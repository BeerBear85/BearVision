import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from bearvision.adapters import BoxJobQueue
from bearvision.contracts import (
    JobResultManifest,
    MediaAsset,
    TagObservation,
    Vector3,
)
from bearvision.edge.job_package import build_edge_job
from bearvision.ports import CapturedMedia, JobQueue
from bearvision.queueing import StoreBackedJobQueue, job_package_files
from bearvision.server import FileSystemJobQueue
from bearvision.simulation import InMemoryJobQueue


NOW = datetime(2026, 8, 28, tzinfo=timezone.utc)
USER_ID = UUID("b10e3918-490c-4a3f-859a-e67c12b66680")
OTHER_USER_ID = UUID("1a082c79-0eb6-4f14-bad2-4612f21889fe")


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
        moving = {
            key: value for key, value in self.files.items() if key.startswith(source_prefix)
        }
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


@dataclass(frozen=True, slots=True)
class QueueHarness:
    name: str
    queue: JobQueue
    restart: Callable[[], JobQueue]
    inject_incomplete_ready: Callable[[str], None]
    package_names: Callable[[str], set[str]]


@pytest.fixture(params=("memory", "filesystem", "box"))
def queue_harness(request, tmp_path: Path) -> QueueHarness:
    if request.param == "memory":
        queue = InMemoryJobQueue()

        def inject(job_id: str) -> None:
            queue.packages[job_id] = {"manifest.json": b"{}"}
            queue.states[job_id] = "ready"

        return QueueHarness(
            name="memory",
            queue=queue,
            restart=lambda: queue,
            inject_incomplete_ready=inject,
            package_names=lambda job_id: set(queue.packages[job_id]),
        )
    if request.param == "filesystem":
        root = tmp_path / "queue"
        queue = FileSystemJobQueue(root)

        def inject(job_id: str) -> None:
            folder = root / "input-queue/ready" / job_id
            folder.mkdir(parents=True)
            (folder / "manifest.json").write_bytes(b"{}")

        def package_names(job_id: str) -> set[str]:
            for parent in ("input-queue/ready", "processing"):
                folder = root / parent / job_id
                if folder.is_dir():
                    return {item.name for item in folder.iterdir()}
            raise FileNotFoundError(job_id)

        return QueueHarness(
            name="filesystem",
            queue=queue,
            restart=lambda: FileSystemJobQueue(root),
            inject_incomplete_ready=inject,
            package_names=package_names,
        )
    handler = MemoryBoxFolders()
    queue = BoxJobQueue(handler, tmp_path / "scratch")

    def inject(job_id: str) -> None:
        handler.files[f"input-queue/ready/{job_id}/manifest.json"] = b"{}"

    def package_names(job_id: str) -> set[str]:
        for parent in ("input-queue/ready", "processing"):
            prefix = f"{parent}/{job_id}/"
            names = {key.removeprefix(prefix) for key in handler.files if key.startswith(prefix)}
            if names:
                return names
        raise FileNotFoundError(job_id)

    return QueueHarness(
        name="box",
        queue=queue,
        restart=lambda: BoxJobQueue(handler, tmp_path / "scratch-restarted"),
        inject_incomplete_ready=inject,
        package_names=package_names,
    )


def sample_job(job_id: str = "job-1"):
    video = CapturedMedia(
        asset=MediaAsset(
            asset_id=f"asset-{job_id}",
            filename="video.mp4",
            content_type="video/mp4",
            size_bytes=5,
            created_at_utc=NOW,
        ),
        content=b"video",
    )
    source = TagObservation(
        tag_id="BearTag-1",
        observed_at_utc=NOW + timedelta(seconds=1),
        observed_at_monotonic_s=1,
        rssi_dbm=-50,
        acceleration_mps2=Vector3(x=1, y=2, z=3),
    )
    manifest, observations = build_edge_job(
        job_id=job_id,
        edge_device_id="edge-1",
        created_at=NOW + timedelta(seconds=5),
        capture_started_at=NOW,
        capture_ended_at=NOW + timedelta(seconds=5),
        clip_start_monotonic_s=0,
        video=video,
        observations=(source,),
    )
    return manifest, video, observations


def processed_result(job_id: str) -> JobResultManifest:
    return JobResultManifest(
        jobId=job_id,
        status="processed",
        processedAt=NOW + timedelta(seconds=6),
        algorithmVersion="contract-v1",
        selectedBearTagId="BearTag-1",
        selectedUserId=USER_ID,
        assignmentId="assignment-1",
        reason="selected contract rider",
    )


def terminal_result(job_id: str, status: str) -> JobResultManifest:
    return JobResultManifest(
        jobId=job_id,
        status=status,
        processedAt=NOW + timedelta(seconds=6),
        algorithmVersion="contract-v1",
        reason=f"contract {status}",
        errorCode=f"CONTRACT_{status.upper()}",
    )


def test_durable_provider_adapters_share_one_queue_lifecycle() -> None:
    lifecycle_methods = (
        "admin_list_jobs",
        "admin_read",
        "admin_download",
        "publish",
        "acquire_next",
        "read",
        "finish",
        "requeue",
        "snapshot",
    )

    for adapter in (FileSystemJobQueue, BoxJobQueue):
        for method in lifecycle_methods:
            assert getattr(adapter, method) is getattr(StoreBackedJobQueue, method)


def test_job_queue_contract_publishes_complete_packages_idempotently_and_resumes_claim(
    queue_harness: QueueHarness,
) -> None:
    queue_harness.inject_incomplete_ready("incomplete")
    assert asyncio.run(queue_harness.queue.acquire_next()) is None
    manifest, video, observations = sample_job()
    expected_files = dict(job_package_files(manifest, video, observations))

    assert asyncio.run(queue_harness.queue.publish(manifest, video, observations))
    assert queue_harness.package_names(manifest.job_id) == set(expected_files)
    assert not asyncio.run(queue_harness.queue.publish(manifest, video, observations))
    assert asyncio.run(queue_harness.queue.acquire_next()) == manifest.job_id
    for filename, expected in expected_files.items():
        assert asyncio.run(queue_harness.queue.read(manifest.job_id, filename)) == expected
    assert asyncio.run(queue_harness.restart().acquire_next()) == manifest.job_id


def test_job_queue_contract_validates_processed_user_and_normalizes_snapshot(
    queue_harness: QueueHarness,
) -> None:
    manifest, video, observations = sample_job()
    assert asyncio.run(queue_harness.queue.publish(manifest, video, observations))
    assert asyncio.run(queue_harness.queue.acquire_next()) == manifest.job_id
    result = processed_result(manifest.job_id)

    with pytest.raises(ValueError, match="requires user id"):
        asyncio.run(queue_harness.queue.finish(manifest.job_id, result))
    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(queue_harness.queue.finish(manifest.job_id, result, OTHER_USER_ID))
    assert queue_harness.queue.snapshot()["counts"]["processing"] == 1

    asyncio.run(queue_harness.queue.finish(manifest.job_id, result, USER_ID))
    assert not asyncio.run(queue_harness.queue.requeue(manifest.job_id))
    snapshot = queue_harness.queue.snapshot()
    assert snapshot["counts"] == {
        "ready": 0,
        "processing": 0,
        "processed": 1,
        "unresolved": 0,
        "failed": 0,
    }
    assert snapshot["jobs"][0]["status"] == "processed"
    assert snapshot["jobs"][0]["userId"] == str(USER_ID)
    assert snapshot["jobs"][0]["selectedUserId"] == str(USER_ID)


@pytest.mark.parametrize("status", ("unresolved", "failed"))
def test_job_queue_contract_requeues_terminal_jobs_without_stale_result(
    queue_harness: QueueHarness,
    status: str,
) -> None:
    manifest, video, observations = sample_job(f"job-{status}")
    assert asyncio.run(queue_harness.queue.publish(manifest, video, observations))
    assert asyncio.run(queue_harness.queue.acquire_next()) == manifest.job_id
    asyncio.run(queue_harness.queue.finish(manifest.job_id, terminal_result(manifest.job_id, status)))
    assert queue_harness.queue.snapshot()["counts"][status] == 1

    assert asyncio.run(queue_harness.queue.requeue(manifest.job_id))
    snapshot = queue_harness.queue.snapshot()
    assert snapshot["counts"]["ready"] == 1
    assert snapshot["jobs"][0] == {"jobId": manifest.job_id, "status": "ready"}
    assert asyncio.run(queue_harness.queue.acquire_next()) == manifest.job_id
    assert asyncio.run(queue_harness.restart().acquire_next()) == manifest.job_id
