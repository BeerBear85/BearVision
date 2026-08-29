"""Box folder-store adapter for the provider-neutral durable job queue."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import tempfile
from typing import Any, TypeVar

from bearvision.queueing import StoreBackedJobQueue

from ._errors import translated_error


T = TypeVar("T")


class BoxQueueStore:
    """Map generic queue folder operations to the synchronous Box handler."""

    def __init__(self, handler: Any, scratch_dir: str | Path) -> None:
        self.handler = handler
        self.scratch_dir = Path(scratch_dir)

    @staticmethod
    def _run(operation: str, action: Callable[[], T]) -> T:
        try:
            return action()
        except (FileNotFoundError, FileExistsError):
            raise
        except Exception as exc:
            raise translated_error(exc, operation) from exc

    def list_folders(self, path: str) -> list[str]:
        return self._run(
            "list Box queue folders", lambda: self.handler.list_folders(path)
        )

    def exists(self, path: str, *, folder: bool) -> bool:
        operation = self.handler.folder_exists if folder else self.handler.file_exists
        return self._run("inspect Box queue path", lambda: operation(path))

    def read(self, path: str) -> bytes:
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.scratch_dir, delete=False
            ) as stream:
                temporary = Path(stream.name)
            self._run(
                "read Box queue file",
                lambda: self.handler.download_file(path, str(temporary)),
            )
            return temporary.read_bytes()
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def download(self, path: str, destination: Path) -> None:
        self._run(
            "download Box queue file",
            lambda: self.handler.download_file(path, str(destination)),
        )

    def write(self, path: str, content: bytes, *, overwrite: bool) -> None:
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.scratch_dir, delete=False
            ) as stream:
                stream.write(content)
                temporary = Path(stream.name)
            self._run(
                "write Box queue file",
                lambda: self.handler.upload_file(
                    str(temporary), path, overwrite
                ),
            )
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def move(self, source: str, destination: str) -> None:
        self._run(
            "move Box queue folder",
            lambda: self.handler.move_folder(source, destination),
        )

    def delete(self, path: str, *, folder: bool) -> None:
        operation = self.handler.delete_folder if folder else self.handler.delete_file
        self._run("delete Box queue path", lambda: operation(path))


class BoxJobQueue(StoreBackedJobQueue):
    """Durable job queue using Box folders as its provider store."""

    def __init__(self, handler: Any, scratch_dir: str | Path) -> None:
        self.handler = handler
        self.scratch_dir = Path(scratch_dir)
        super().__init__(
            BoxQueueStore(handler, self.scratch_dir),
            unique_upload_folders=False,
            retain_failed_uploads=True,
        )
