"""Local folder-store adapter for the provider-neutral durable job queue."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from uuid import uuid4

from bearvision.queueing import StoreBackedJobQueue


class FileSystemQueueStore:
    """Map generic queue folder operations to one local filesystem root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, path: str) -> Path:
        return self.root / Path(path)

    def list_folders(self, path: str) -> list[str]:
        folder = self._path(path)
        if not folder.exists():
            return []
        return [item.name for item in folder.iterdir() if item.is_dir()]

    def exists(self, path: str, *, folder: bool) -> bool:
        target = self._path(path)
        return target.is_dir() if folder else target.is_file()

    def read(self, path: str) -> bytes:
        return self._path(path).read_bytes()

    def download(self, path: str, destination: Path) -> None:
        shutil.copyfile(self._path(path), destination)

    def write(self, path: str, content: bytes, *, overwrite: bool) -> None:
        destination = self._path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            raise FileExistsError(path)
        temporary = destination.with_name(
            f".{destination.name}.{uuid4().hex}.tmp"
        )
        try:
            temporary.write_bytes(content)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def move(self, source: str, destination: str) -> None:
        target = self._path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self._path(source), target)

    def delete(self, path: str, *, folder: bool) -> None:
        target = self._path(path)
        if folder:
            if target.exists():
                shutil.rmtree(target)
            return
        target.unlink(missing_ok=True)


class FileSystemJobQueue(StoreBackedJobQueue):
    """Durable job queue using local folders as its provider store."""

    STATES = ("ready", "processing", "processed", "unresolved", "failed")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        for state in (
            "input-queue/uploading",
            "input-queue/ready",
            *self.STATES[1:],
        ):
            (self.root / state).mkdir(parents=True, exist_ok=True)
        super().__init__(
            FileSystemQueueStore(self.root),
            unique_upload_folders=True,
            retain_failed_uploads=False,
        )
