"""Provider-neutral job package rules and durable queue lifecycle."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Protocol
from typing import Any
from uuid import UUID, uuid4

from bearvision.contracts import (
    BearTagJobObservation,
    EdgeJobManifest,
    JobResultManifest,
)
from bearvision.contracts.identity import (
    user_id_from_storage_folder,
    user_storage_folder,
)
from bearvision.ports import CapturedMedia


QUEUE_STATES = ("ready", "processing", "processed", "unresolved", "failed")


class QueueFolderStore(Protocol):
    """Small provider seam used by the durable job queue lifecycle."""

    def list_folders(self, path: str) -> list[str]: ...

    def exists(self, path: str, *, folder: bool) -> bool: ...

    def read(self, path: str) -> bytes: ...

    def download(self, path: str, destination: Path) -> None: ...

    def write(self, path: str, content: bytes, *, overwrite: bool) -> None: ...

    def move(self, source: str, destination: str) -> None: ...

    def delete(self, path: str, *, folder: bool) -> None: ...


def media_bytes(media: CapturedMedia) -> bytes:
    if media.content is not None:
        return media.content
    assert media.local_path is not None
    return media.local_path.read_bytes()


def serialize_manifest(manifest: EdgeJobManifest) -> bytes:
    return (manifest.model_dump_json(by_alias=True, indent=2) + "\n").encode()


def serialize_observations(
    observations: tuple[BearTagJobObservation, ...],
) -> bytes:
    return "".join(
        json.dumps(item.model_dump(mode="json", by_alias=True), separators=(",", ":"))
        + "\n"
        for item in observations
    ).encode()


def serialize_result(result: JobResultManifest) -> bytes:
    return (result.model_dump_json(by_alias=True, indent=2) + "\n").encode()


def job_package_files(
    manifest: EdgeJobManifest,
    video: CapturedMedia,
    observations: tuple[BearTagJobObservation, ...],
) -> tuple[tuple[str, bytes], ...]:
    """Return a complete package in publication order, with READY strictly last."""

    return (
        (manifest.video.filename, media_bytes(video)),
        ("manifest.json", serialize_manifest(manifest)),
        (manifest.observations_filename, serialize_observations(observations)),
        ("READY", b""),
    )


def validate_result_destination(
    result: JobResultManifest,
    user_id: UUID | None,
) -> None:
    if result.status == "processed":
        if user_id is None:
            raise ValueError("processed result requires user id")
        if result.selected_user_id != user_id:
            raise ValueError("processed result user id does not match destination")
    elif user_id is not None:
        raise ValueError("non-processed result must not have a destination user id")


def normalize_queue_snapshot(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize provider-discovered jobs to one ordered administrative view."""

    normalized: list[dict[str, Any]] = []
    for source in jobs:
        item = dict(source)
        job_id = item.get("jobId")
        status = item.get("status")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("queue snapshot job requires jobId")
        if status not in QUEUE_STATES:
            raise ValueError(f"invalid queue snapshot status: {status!r}")
        normalized.append(item)
    state_order = {state: index for index, state in enumerate(QUEUE_STATES)}
    normalized.sort(key=lambda item: (state_order[item["status"]], item["jobId"]))
    return {
        "counts": {
            state: sum(item["status"] == state for item in normalized)
            for state in QUEUE_STATES
        },
        "jobs": normalized,
    }


class StoreBackedJobQueue:
    """Own every durable queue transition above a provider-neutral folder store."""

    def __init__(
        self,
        store: QueueFolderStore,
        *,
        unique_upload_folders: bool,
        retain_failed_uploads: bool,
    ) -> None:
        self._store = store
        self._unique_upload_folders = unique_upload_folders
        self._retain_failed_uploads = retain_failed_uploads

    @staticmethod
    def _validate_job_id(job_id: str) -> None:
        if not job_id or Path(job_id).name != job_id:
            raise ValueError("invalid job id")

    @staticmethod
    def _validate_filename(filename: str) -> None:
        if Path(filename).name != filename:
            raise ValueError("job filename must not contain a path")

    def _processed_users(self) -> list[tuple[str, UUID]]:
        result: list[tuple[str, UUID]] = []
        for folder in self._store.list_folders("processed"):
            try:
                result.append((folder, user_id_from_storage_folder(folder)))
            except ValueError:
                continue
        return sorted(result)

    def _location(self, job_id: str) -> tuple[str, str, str | None]:
        self._validate_job_id(job_id)
        for status, path in (
            ("ready", f"input-queue/ready/{job_id}"),
            ("processing", f"processing/{job_id}"),
            ("unresolved", f"unresolved/{job_id}"),
            ("failed", f"failed/{job_id}"),
        ):
            if self._store.exists(path, folder=True):
                return status, path, None
        for user_folder, user_id in self._processed_users():
            path = f"processed/{user_folder}/{job_id}"
            if self._store.exists(path, folder=True):
                return "processed", path, str(user_id)
        raise FileNotFoundError(job_id)

    def admin_list_jobs(self) -> list[dict[str, str]]:
        """List lightweight job locations without loading job payloads."""

        result: list[dict[str, str]] = []
        for status, path in (
            ("ready", "input-queue/ready"),
            ("processing", "processing"),
            ("unresolved", "unresolved"),
            ("failed", "failed"),
        ):
            result.extend(
                {"jobId": job_id, "status": status}
                for job_id in self._store.list_folders(path)
            )
        for user_folder, user_id in self._processed_users():
            result.extend(
                {"jobId": job_id, "status": "processed", "userId": str(user_id)}
                for job_id in self._store.list_folders(f"processed/{user_folder}")
            )
        return sorted(result, key=lambda item: (item["status"], item["jobId"]))

    async def admin_read(self, job_id: str, filename: str) -> bytes:
        self._validate_filename(filename)
        _, folder, _ = await asyncio.to_thread(self._location, job_id)
        return await asyncio.to_thread(self._store.read, f"{folder}/{filename}")

    async def admin_download(
        self, job_id: str, filename: str, destination: Path
    ) -> None:
        self._validate_filename(filename)
        _, folder, _ = await asyncio.to_thread(self._location, job_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            self._store.download, f"{folder}/{filename}", destination
        )

    async def _exists_anywhere(self, job_id: str) -> bool:
        for path in (
            f"input-queue/ready/{job_id}",
            f"processing/{job_id}",
            f"unresolved/{job_id}",
            f"failed/{job_id}",
        ):
            if await asyncio.to_thread(self._store.exists, path, folder=True):
                return True
        users = await asyncio.to_thread(self._processed_users)
        for user_folder, _ in users:
            if await asyncio.to_thread(
                self._store.exists,
                f"processed/{user_folder}/{job_id}",
                folder=True,
            ):
                return True
        return False

    def _upload_folder(self, job_id: str) -> str:
        if not self._unique_upload_folders:
            return f"input-queue/uploading/{job_id}"
        return f"input-queue/uploading/.{job_id}.{uuid4().hex}.tmp"

    async def _delete_folder_if_present(self, path: str) -> None:
        if await asyncio.to_thread(self._store.exists, path, folder=True):
            await asyncio.to_thread(self._store.delete, path, folder=True)

    async def publish(
        self,
        manifest: EdgeJobManifest,
        video: CapturedMedia,
        observations: tuple[BearTagJobObservation, ...],
    ) -> bool:
        self._validate_job_id(manifest.job_id)
        if await self._exists_anywhere(manifest.job_id):
            return False
        uploading = self._upload_folder(manifest.job_id)
        try:
            for filename, content in job_package_files(manifest, video, observations):
                await asyncio.to_thread(
                    self._store.write,
                    f"{uploading}/{filename}",
                    content,
                    overwrite=True,
                )
            if await self._exists_anywhere(manifest.job_id):
                await self._delete_folder_if_present(uploading)
                return False
            await asyncio.to_thread(
                self._store.move,
                uploading,
                f"input-queue/ready/{manifest.job_id}",
            )
            return True
        except Exception:
            if not self._retain_failed_uploads:
                await self._delete_folder_if_present(uploading)
            raise

    async def acquire_next(self) -> str | None:
        processing = sorted(
            await asyncio.to_thread(self._store.list_folders, "processing")
        )
        if processing:
            return processing[0]
        ready = sorted(
            await asyncio.to_thread(self._store.list_folders, "input-queue/ready")
        )
        for job_id in ready:
            ready_file = f"input-queue/ready/{job_id}/READY"
            if not await asyncio.to_thread(
                self._store.exists, ready_file, folder=False
            ):
                continue
            try:
                await asyncio.to_thread(
                    self._store.move,
                    f"input-queue/ready/{job_id}",
                    f"processing/{job_id}",
                )
            except (FileNotFoundError, FileExistsError):
                continue
            return job_id
        return None

    async def read(self, job_id: str, filename: str) -> bytes:
        self._validate_job_id(job_id)
        self._validate_filename(filename)
        return await asyncio.to_thread(
            self._store.read, f"processing/{job_id}/{filename}"
        )

    async def finish(
        self, job_id: str, result: JobResultManifest, user_id: UUID | None = None
    ) -> None:
        self._validate_job_id(job_id)
        validate_result_destination(result, user_id)
        source = f"processing/{job_id}"
        if result.status == "processed":
            assert user_id is not None
            destination = f"processed/{user_storage_folder(user_id)}/{job_id}"
        else:
            destination = f"{result.status}/{job_id}"
        if await asyncio.to_thread(self._store.exists, destination, folder=True):
            await self._delete_folder_if_present(source)
            return
        if not await asyncio.to_thread(self._store.exists, source, folder=True):
            raise FileNotFoundError(job_id)
        await asyncio.to_thread(
            self._store.write,
            f"{source}/result.json",
            serialize_result(result),
            overwrite=True,
        )
        await asyncio.to_thread(self._store.move, source, destination)

    async def requeue(self, job_id: str) -> bool:
        self._validate_job_id(job_id)
        for state in ("failed", "unresolved"):
            source = f"{state}/{job_id}"
            if not await asyncio.to_thread(self._store.exists, source, folder=True):
                continue
            result_file = f"{source}/result.json"
            if await asyncio.to_thread(
                self._store.exists, result_file, folder=False
            ):
                await asyncio.to_thread(
                    self._store.delete, result_file, folder=False
                )
            await asyncio.to_thread(
                self._store.move, source, f"input-queue/ready/{job_id}"
            )
            return True
        return False

    def _result_details(self, folder: str) -> dict[str, Any]:
        result_file = f"{folder}/result.json"
        if not self._store.exists(result_file, folder=False):
            return {}
        return json.loads(self._store.read(result_file))

    def _jobs(self, status: str, folder: str) -> list[dict[str, Any]]:
        return [
            {
                "jobId": job_id,
                "status": status,
                **self._result_details(f"{folder}/{job_id}"),
            }
            for job_id in self._store.list_folders(folder)
        ]

    def snapshot(self) -> dict[str, Any]:
        jobs = (
            self._jobs("ready", "input-queue/ready")
            + self._jobs("processing", "processing")
            + self._jobs("unresolved", "unresolved")
            + self._jobs("failed", "failed")
        )
        for user_folder, user_id in self._processed_users():
            jobs.extend(
                {
                    "jobId": job_id,
                    "status": "processed",
                    "userId": str(user_id),
                    **self._result_details(
                        f"processed/{user_folder}/{job_id}"
                    ),
                }
                for job_id in self._store.list_folders(
                    f"processed/{user_folder}"
                )
            )
        return normalize_queue_snapshot(jobs)
