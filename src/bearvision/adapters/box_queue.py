"""Durable Box-backed implementation of the job queue port."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
from typing import Any
from uuid import UUID

from bearvision.contracts import BearTagJobObservation, EdgeJobManifest, JobResultManifest
from bearvision.contracts.identity import user_id_from_storage_folder, user_storage_folder
from bearvision.ports import CapturedMedia
from bearvision.queueing import (
    job_package_files,
    normalize_queue_snapshot,
    serialize_result,
    validate_result_destination,
)

from ._errors import translated_error


class BoxJobQueue:
    def __init__(self, handler: Any, scratch_dir: str | Path) -> None:
        self.handler = handler
        self.scratch_dir = Path(scratch_dir)

    def _paths(self, job_id: str) -> tuple[str, ...]:
        return (
            f"input-queue/ready/{job_id}",
            f"processing/{job_id}",
            f"unresolved/{job_id}",
            f"failed/{job_id}",
        )

    def _admin_location(self, job_id: str) -> tuple[str, str, str | None]:
        if not job_id or Path(job_id).name != job_id:
            raise ValueError("invalid job id")
        for status, path in (
            ("ready", f"input-queue/ready/{job_id}"),
            ("processing", f"processing/{job_id}"),
            ("unresolved", f"unresolved/{job_id}"),
            ("failed", f"failed/{job_id}"),
        ):
            if self.handler.folder_exists(path):
                return status, path, None
        for user_folder, user_id in self._processed_users():
            path = f"processed/{user_folder}/{job_id}"
            if self.handler.folder_exists(path):
                return "processed", path, str(user_id)
        raise FileNotFoundError(job_id)

    def _processed_users(self) -> list[tuple[str, UUID]]:
        result: list[tuple[str, UUID]] = []
        for folder in self.handler.list_folders("processed"):
            try:
                result.append((folder, user_id_from_storage_folder(folder)))
            except ValueError:
                continue
        return result

    def admin_list_jobs(self) -> list[dict[str, str]]:
        """List job identities and locations without downloading manifests."""

        result: list[dict[str, str]] = []
        for status, path in (
            ("ready", "input-queue/ready"),
            ("processing", "processing"),
            ("unresolved", "unresolved"),
            ("failed", "failed"),
        ):
            result.extend(
                {"jobId": job_id, "status": status}
                for job_id in self.handler.list_folders(path)
            )
        for user_folder, user_id in self._processed_users():
            result.extend(
                {"jobId": job_id, "status": "processed", "userId": str(user_id)}
                for job_id in self.handler.list_folders(f"processed/{user_folder}")
            )
        return result

    async def admin_read(self, job_id: str, filename: str) -> bytes:
        if Path(filename).name != filename:
            raise ValueError("job filename must not contain a path")
        _, folder, _ = await asyncio.to_thread(self._admin_location, job_id)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.scratch_dir, delete=False) as stream:
                temporary = Path(stream.name)
            await asyncio.to_thread(
                self.handler.download_file, f"{folder}/{filename}", str(temporary)
            )
            return temporary.read_bytes()
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise translated_error(exc, "read archived Box job file") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    async def admin_download(
        self, job_id: str, filename: str, destination: Path
    ) -> None:
        if Path(filename).name != filename:
            raise ValueError("job filename must not contain a path")
        _, folder, _ = await asyncio.to_thread(self._admin_location, job_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            await asyncio.to_thread(
                self.handler.download_file, f"{folder}/{filename}", str(destination)
            )
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise translated_error(exc, "download archived Box job file") from exc

    async def _exists_anywhere(self, job_id: str) -> bool:
        for path in self._paths(job_id):
            if await asyncio.to_thread(self.handler.folder_exists, path):
                return True
        users = await asyncio.to_thread(self._processed_users)
        for user_folder, _ in users:
            if await asyncio.to_thread(
                self.handler.folder_exists, f"processed/{user_folder}/{job_id}"
            ):
                return True
        return False

    async def _upload_bytes(
        self, content: bytes, destination: str, *, overwrite: bool = False
    ) -> None:
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.scratch_dir, delete=False) as stream:
                stream.write(content)
                temporary = Path(stream.name)
            await asyncio.to_thread(
                self.handler.upload_file, str(temporary), destination, overwrite
            )
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    async def publish(
        self,
        manifest: EdgeJobManifest,
        video: CapturedMedia,
        observations: tuple[BearTagJobObservation, ...],
    ) -> bool:
        try:
            if await self._exists_anywhere(manifest.job_id):
                return False
            prefix = f"input-queue/uploading/{manifest.job_id}"
            # A prior transient failure may have left an incomplete uploading
            # folder. Rewriting its deterministic files safely resumes it.
            for filename, content in job_package_files(manifest, video, observations):
                await self._upload_bytes(
                    content,
                    f"{prefix}/{filename}",
                    overwrite=True,
                )
            await asyncio.to_thread(
                self.handler.move_folder,
                prefix,
                f"input-queue/ready/{manifest.job_id}",
            )
            return True
        except Exception as exc:
            raise translated_error(exc, "publish complete job to Box") from exc

    async def acquire_next(self) -> str | None:
        try:
            processing = sorted(await asyncio.to_thread(self.handler.list_folders, "processing"))
            if processing:
                return processing[0]
            ready = sorted(await asyncio.to_thread(self.handler.list_folders, "input-queue/ready"))
            for job_id in ready:
                if not await asyncio.to_thread(
                    self.handler.file_exists, f"input-queue/ready/{job_id}/READY"
                ):
                    continue
                await asyncio.to_thread(
                    self.handler.move_folder,
                    f"input-queue/ready/{job_id}",
                    f"processing/{job_id}",
                )
                return job_id
            return None
        except Exception as exc:
            raise translated_error(exc, "claim Box job") from exc

    async def read(self, job_id: str, filename: str) -> bytes:
        if Path(filename).name != filename:
            raise ValueError("job filename must not contain a path")
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.scratch_dir, delete=False) as stream:
                temporary = Path(stream.name)
            await asyncio.to_thread(
                self.handler.download_file,
                f"processing/{job_id}/{filename}",
                str(temporary),
            )
            return temporary.read_bytes()
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise translated_error(exc, "read Box job file") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    async def finish(
        self, job_id: str, result: JobResultManifest, user_id: UUID | None = None
    ) -> None:
        validate_result_destination(result, user_id)
        try:
            await self._upload_bytes(
                serialize_result(result),
                f"processing/{job_id}/result.json",
                overwrite=True,
            )
            if result.status == "processed":
                assert user_id is not None
                destination = f"processed/{user_storage_folder(user_id)}/{job_id}"
            else:
                destination = f"{result.status}/{job_id}"
            await asyncio.to_thread(
                self.handler.move_folder, f"processing/{job_id}", destination
            )
        except Exception as exc:
            raise translated_error(exc, "finish Box job") from exc

    async def requeue(self, job_id: str) -> bool:
        try:
            for state in ("failed", "unresolved"):
                source = f"{state}/{job_id}"
                if await asyncio.to_thread(self.handler.folder_exists, source):
                    await asyncio.to_thread(
                        self.handler.delete_file, f"{source}/result.json"
                    )
                    await asyncio.to_thread(
                        self.handler.move_folder, source, f"input-queue/ready/{job_id}"
                    )
                    return True
            return False
        except Exception as exc:
            raise translated_error(exc, "requeue Box job") from exc

    def _result_details(self, folder: str) -> dict:
        remote = f"{folder}/result.json"
        if not self.handler.file_exists(remote):
            return {}
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.scratch_dir, delete=False) as stream:
                temporary = Path(stream.name)
            self.handler.download_file(remote, str(temporary))
            return json.loads(temporary.read_text(encoding="utf-8"))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def snapshot(self) -> dict:
        try:
            ready = self.handler.list_folders("input-queue/ready")
            processing = self.handler.list_folders("processing")
            unresolved = self.handler.list_folders("unresolved")
            failed = self.handler.list_folders("failed")
            processed = [
                (user_folder, user_id, job_id)
                for user_folder, user_id in self._processed_users()
                for job_id in self.handler.list_folders(f"processed/{user_folder}")
            ]
            jobs = (
                [{"jobId": item, "status": "ready"} for item in ready]
                + [{"jobId": item, "status": "processing"} for item in processing]
                + [
                    {
                        "jobId": job_id,
                        "status": "processed",
                        "userId": str(user_id),
                        **self._result_details(f"processed/{user_folder}/{job_id}"),
                    }
                    for user_folder, user_id, job_id in processed
                ]
                + [
                    {
                        "jobId": item,
                        "status": "unresolved",
                        **self._result_details(f"unresolved/{item}"),
                    }
                    for item in unresolved
                ]
                + [
                    {
                        "jobId": item,
                        "status": "failed",
                        **self._result_details(f"failed/{item}"),
                    }
                    for item in failed
                ]
            )
            return normalize_queue_snapshot(jobs)
        except Exception as exc:
            raise translated_error(exc, "inspect Box queue") from exc
