"""Typed ports implemented by both production and simulation adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from bearvision.contracts import (
    CaptureRequest,
    PersonDetection,
    StorageReceipt,
    TagObservation,
    TagRegistryEntry,
)

from .models import CapturedMedia, ExtractedClip, VideoFrame


@runtime_checkable
class Clock(Protocol):
    def utc_now(self) -> datetime: ...

    def monotonic(self) -> float: ...

    async def sleep(self, delay_s: float) -> None: ...


@runtime_checkable
class Camera(Protocol):
    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    async def start_preview(self) -> str: ...

    async def stop_preview(self) -> None: ...

    async def capture(self, request: CaptureRequest) -> CapturedMedia: ...


@runtime_checkable
class VideoClipper(Protocol):
    async def extract(
        self,
        source: Path,
        destination: Path,
        *,
        start_s: float,
        duration_s: float,
    ) -> ExtractedClip: ...


@runtime_checkable
class FrameSource(Protocol):
    """Turn a camera preview endpoint into timestamped video frames."""

    async def open(self, preview_source: str) -> None: ...

    async def close(self) -> None: ...

    def frames(self) -> AsyncIterator[VideoFrame]: ...


@runtime_checkable
class TagScanner(Protocol):
    def observations(self) -> AsyncIterator[TagObservation]: ...


@runtime_checkable
class Detector(Protocol):
    async def detect(self, frame: VideoFrame) -> tuple[PersonDetection, ...]: ...


@runtime_checkable
class Storage(Protocol):
    async def upload(
        self, media: CapturedMedia, object_key: str, *, overwrite: bool = False
    ) -> StorageReceipt: ...

    async def download(self, object_key: str) -> bytes: ...

    async def delete(self, object_key: str) -> None: ...


@runtime_checkable
class TagRegistry(Protocol):
    def resolve(self, tag_id: str) -> TagRegistryEntry | None: ...

    def entries(self) -> Iterable[TagRegistryEntry]: ...
