"""Storage port adapter for the existing BoxHandler."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
import tempfile
from typing import Any

from bearvision.contracts import StorageReceipt
from bearvision.ports import CapturedMedia

from ._errors import translated_error


class BoxStorageAdapter:
    def __init__(self, handler: Any, clock: Any, scratch_dir: str | Path) -> None:
        self.handler = handler
        self.clock = clock
        self.scratch_dir = Path(scratch_dir)
        self._receipts: dict[tuple[str, str], StorageReceipt] = {}

    async def upload(
        self, media: CapturedMedia, object_key: str, *, overwrite: bool = False
    ) -> StorageReceipt:
        cache_key = (media.asset.asset_id, object_key)
        if cache_key in self._receipts:
            return self._receipts[cache_key]
        temporary_path: Path | None = None
        try:
            if media.content is not None:
                self.scratch_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    dir=self.scratch_dir, suffix=Path(media.asset.filename).suffix, delete=False
                ) as stream:
                    stream.write(media.content)
                    temporary_path = Path(stream.name)
                source = temporary_path
                content = media.content
            else:
                source = media.local_path
                content = source.read_bytes()
            await asyncio.to_thread(
                self.handler.upload_file, str(source), object_key, overwrite
            )
            receipt = StorageReceipt(
                asset_id=media.asset.asset_id,
                object_key=object_key,
                stored_at_utc=self.clock.utc_now(),
                checksum_sha256=hashlib.sha256(content).hexdigest(),
            )
            self._receipts[cache_key] = receipt
            return receipt
        except Exception as exc:
            raise translated_error(exc, "upload media to Box") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    async def download(self, object_key: str) -> bytes:
        temporary_path: Path | None = None
        try:
            self.scratch_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.scratch_dir, delete=False) as stream:
                temporary_path = Path(stream.name)
            await asyncio.to_thread(self.handler.download_file, object_key, str(temporary_path))
            return temporary_path.read_bytes()
        except Exception as exc:
            raise translated_error(exc, "download media from Box") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    async def delete(self, object_key: str) -> None:
        try:
            await asyncio.to_thread(self.handler.delete_file, object_key)
            for key in tuple(self._receipts):
                if key[1] == object_key:
                    del self._receipts[key]
        except Exception as exc:
            raise translated_error(exc, "delete media from Box") from exc
