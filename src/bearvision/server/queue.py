"""Provider-neutral job queue and local persistent adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from uuid import UUID, uuid4

from bearvision.contracts import BearTagJobObservation, EdgeJobManifest, JobResultManifest
from bearvision.contracts.identity import user_id_from_storage_folder, user_storage_folder
from bearvision.ports import CapturedMedia
from bearvision.queueing import (
    job_package_files,
    normalize_queue_snapshot,
    serialize_result,
    validate_result_destination,
)


class FileSystemJobQueue:
    """Filesystem simulation with the same durable state transitions as Box."""

    STATES = ("ready", "processing", "processed", "unresolved", "failed")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        for state in ("input-queue/uploading", "input-queue/ready", *self.STATES[1:]):
            (self.root / state).mkdir(parents=True, exist_ok=True)

    def _terminal_or_active_path(self, job_id: str) -> Path | None:
        if not job_id or Path(job_id).name != job_id:
            raise ValueError("invalid job id")
        candidates = [
            self.root / "input-queue/ready" / job_id,
            self.root / "processing" / job_id,
            self.root / "unresolved" / job_id,
            self.root / "failed" / job_id,
        ]
        candidates.extend((self.root / "processed").glob(f"*/{job_id}"))
        return next((item for item in candidates if item.exists()), None)

    def admin_list_jobs(self) -> list[dict[str, str]]:
        """List lightweight job locations without loading job payloads."""

        result: list[dict[str, str]] = []
        locations = (
            ("ready", self.root / "input-queue/ready"),
            ("processing", self.root / "processing"),
            ("unresolved", self.root / "unresolved"),
            ("failed", self.root / "failed"),
        )
        for status, root in locations:
            if root.exists():
                result.extend(
                    {"jobId": item.name, "status": status}
                    for item in root.iterdir()
                    if item.is_dir() and not item.name.startswith(".")
                )
        processed = self.root / "processed"
        if processed.exists():
            for user_folder in processed.iterdir():
                if user_folder.is_dir():
                    try:
                        user_id = user_id_from_storage_folder(user_folder.name)
                    except ValueError:
                        continue
                    result.extend(
                        {
                            "jobId": item.name,
                            "status": "processed",
                            "userId": str(user_id),
                        }
                        for item in user_folder.iterdir()
                        if item.is_dir()
                    )
        return result

    async def admin_read(self, job_id: str, filename: str) -> bytes:
        if Path(filename).name != filename:
            raise ValueError("job filename must not contain a path")
        folder = self._terminal_or_active_path(job_id)
        if folder is None:
            raise FileNotFoundError(job_id)
        return (folder / filename).read_bytes()

    async def admin_download(
        self, job_id: str, filename: str, destination: Path
    ) -> None:
        if Path(filename).name != filename:
            raise ValueError("job filename must not contain a path")
        folder = self._terminal_or_active_path(job_id)
        if folder is None:
            raise FileNotFoundError(job_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / filename, destination)

    async def publish(
        self,
        manifest: EdgeJobManifest,
        video: CapturedMedia,
        observations: tuple[BearTagJobObservation, ...],
    ) -> bool:
        if self._terminal_or_active_path(manifest.job_id) is not None:
            return False
        uploading = self.root / "input-queue/uploading"
        temporary = uploading / f".{manifest.job_id}.{uuid4().hex}.tmp"
        temporary.mkdir(parents=True)
        try:
            for filename, content in job_package_files(manifest, video, observations):
                (temporary / filename).write_bytes(content)
            destination = self.root / "input-queue/ready" / manifest.job_id
            if self._terminal_or_active_path(manifest.job_id) is not None:
                return False
            os.replace(temporary, destination)
            return True
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    async def acquire_next(self) -> str | None:
        processing = sorted(item.name for item in (self.root / "processing").iterdir() if item.is_dir())
        if processing:
            return processing[0]
        ready_root = self.root / "input-queue/ready"
        for source in sorted(item for item in ready_root.iterdir() if item.is_dir()):
            if not (source / "READY").is_file():
                continue
            destination = self.root / "processing" / source.name
            try:
                os.replace(source, destination)
            except (FileNotFoundError, FileExistsError):
                continue
            return source.name
        return None

    async def read(self, job_id: str, filename: str) -> bytes:
        if Path(filename).name != filename:
            raise ValueError("job filename must not contain a path")
        return (self.root / "processing" / job_id / filename).read_bytes()

    async def finish(
        self, job_id: str, result: JobResultManifest, user_id: UUID | None = None
    ) -> None:
        validate_result_destination(result, user_id)
        source = self.root / "processing" / job_id
        temporary = source / ".result.json.tmp"
        temporary.write_bytes(serialize_result(result))
        os.replace(temporary, source / "result.json")
        if result.status == "processed":
            assert user_id is not None
            destination = self.root / "processed" / user_storage_folder(user_id) / job_id
        else:
            destination = self.root / result.status / job_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(source)
            return
        os.replace(source, destination)

    async def requeue(self, job_id: str) -> bool:
        for state in ("failed", "unresolved"):
            source = self.root / state / job_id
            if source.is_dir():
                (source / "result.json").unlink(missing_ok=True)
                os.replace(source, self.root / "input-queue/ready" / job_id)
                return True
        return False

    def snapshot(self) -> dict:
        def jobs(path: Path) -> list[dict[str, str]]:
            result = []
            for item in sorted(path.iterdir()) if path.exists() else []:
                if item.is_dir():
                    details: dict[str, str] = {"jobId": item.name}
                    result_file = item / "result.json"
                    if result_file.exists():
                        details.update(json.loads(result_file.read_text(encoding="utf-8")))
                    result.append(details)
            return result

        ready = jobs(self.root / "input-queue/ready")
        processing = jobs(self.root / "processing")
        unresolved = jobs(self.root / "unresolved")
        failed = jobs(self.root / "failed")
        processed: list[dict[str, str]] = []
        for user_folder in sorted((self.root / "processed").iterdir()):
            if user_folder.is_dir():
                try:
                    user_id = user_id_from_storage_folder(user_folder.name)
                except ValueError:
                    continue
                processed.extend(
                    {**item, "userId": str(user_id)} for item in jobs(user_folder)
                )
        for state, state_jobs in (
            ("ready", ready),
            ("processing", processing),
            ("processed", processed),
            ("unresolved", unresolved),
            ("failed", failed),
        ):
            for item in state_jobs:
                item.setdefault("status", state)
        return normalize_queue_snapshot(
            ready + processing + processed + unresolved + failed
        )
