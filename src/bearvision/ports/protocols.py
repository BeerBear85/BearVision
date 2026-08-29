"""Typed ports implemented by both production and simulation adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import UUID

from bearvision.contracts import (
    BearTagJobObservation,
    CaptureRequest,
    EdgeJobManifest,
    JobResultManifest,
    PersonDetection,
    StorageReceipt,
    TagObservation,
)

from .models import CapturedClip, CapturedMedia, ExtractedClip, PreparedClip, VideoFrame


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

    async def capture(self, request: CaptureRequest) -> CapturedClip: ...


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
class MediaProbe(Protocol):
    """Read validated timing metadata from a local media file."""

    async def duration(self, source: Path) -> float: ...


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
class ClipProcessor(Protocol):
    """Reduce a captured clip before it enters the cloud job package."""

    async def process(self, media: CapturedMedia) -> PreparedClip: ...


@runtime_checkable
class Storage(Protocol):
    async def upload(
        self, media: CapturedMedia, object_key: str, *, overwrite: bool = False
    ) -> StorageReceipt: ...

    async def download(self, object_key: str) -> bytes: ...

    async def delete(self, object_key: str) -> None: ...


@runtime_checkable
class JobQueue(Protocol):
    """Durable provider-neutral state transitions for cloud job packages."""

    async def publish(
        self,
        manifest: EdgeJobManifest,
        video: CapturedMedia,
        observations: tuple[BearTagJobObservation, ...],
    ) -> bool: ...

    async def acquire_next(self) -> str | None: ...

    async def read(self, job_id: str, filename: str) -> bytes: ...

    async def finish(
        self, job_id: str, result: JobResultManifest, user_id: UUID | None = None
    ) -> None: ...

    async def requeue(self, job_id: str) -> bool: ...

    def snapshot(self) -> dict: ...


@runtime_checkable
class ManagedJobQueue(JobQueue, Protocol):
    """Job queue capabilities used by the local administrative read model."""

    def admin_list_jobs(self) -> list[dict[str, str]]: ...

    async def admin_read(self, job_id: str, filename: str) -> bytes: ...

    async def admin_download(self, job_id: str, filename: str, destination: Path) -> None: ...
