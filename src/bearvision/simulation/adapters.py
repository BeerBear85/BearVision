"""Deterministic component adapters for behavioural simulations."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

from bearvision.contracts import (
    CaptureRequest,
    MediaAsset,
    PersonDetection,
    StorageReceipt,
    TagObservation,
    TagRegistryEntry,
)
from bearvision.ports import CapturedMedia, ComponentUnavailable, VideoFrame


class VirtualClock:
    def __init__(self, start_utc: datetime | None = None) -> None:
        self.start_utc = start_utc or datetime(2026, 1, 1, tzinfo=timezone.utc)
        if self.start_utc.tzinfo is None or self.start_utc.utcoffset() is None:
            raise ValueError("start_utc must be timezone-aware")
        self.elapsed_s = 0.0

    def utc_now(self) -> datetime:
        return self.start_utc + timedelta(seconds=self.elapsed_s)

    def monotonic(self) -> float:
        return self.elapsed_s

    async def sleep(self, delay_s: float) -> None:
        self.advance_by(delay_s)

    def advance_to(self, target_s: float) -> None:
        if target_s < self.elapsed_s:
            raise ValueError("virtual clock cannot move backwards")
        self.elapsed_s = target_s

    def advance_by(self, delay_s: float) -> None:
        if delay_s < 0:
            raise ValueError("delay_s must not be negative")
        self.elapsed_s += delay_s


class SimulatedCamera:
    def __init__(self, clock: VirtualClock, *, fail_capture: bool = False) -> None:
        self.clock = clock
        self.fail_capture = fail_capture
        self.connected = False
        self.previewing = False
        self.captures: dict[str, CapturedMedia] = {}

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.previewing = False
        self.connected = False

    async def start_preview(self) -> str:
        if not self.connected:
            raise ComponentUnavailable("simulated camera is disconnected")
        self.previewing = True
        return "sim://camera/preview"

    async def stop_preview(self) -> None:
        self.previewing = False

    async def capture(self, request: CaptureRequest) -> CapturedMedia:
        if not self.connected:
            raise ComponentUnavailable("simulated camera is disconnected")
        if self.fail_capture:
            raise ComponentUnavailable("injected camera capture failure")
        if request.request_id not in self.captures:
            content = f"bearvision-simulated-clip:{request.request_id}".encode()
            self.captures[request.request_id] = CapturedMedia(
                asset=MediaAsset(
                    asset_id=f"asset-{request.request_id}",
                    filename=f"{request.request_id}.mp4",
                    content_type="video/mp4",
                    size_bytes=len(content),
                    created_at_utc=self.clock.utc_now(),
                ),
                content=content,
            )
        return self.captures[request.request_id]


class SimulatedTagScanner:
    def __init__(self, observations: Iterable[TagObservation]) -> None:
        self._observations = tuple(observations)

    async def observations(self):
        for observation in self._observations:
            yield observation


class SimulatedDetector:
    def __init__(self, detections: dict[str, tuple[PersonDetection, ...]]) -> None:
        self._detections = dict(detections)

    async def detect(self, frame: VideoFrame) -> tuple[PersonDetection, ...]:
        return self._detections.get(frame.frame_id, ())


class InMemoryStorage:
    def __init__(self, clock: VirtualClock, *, fail_upload: bool = False) -> None:
        self.clock = clock
        self.fail_upload = fail_upload
        self.objects: dict[str, tuple[bytes, StorageReceipt]] = {}

    async def upload(
        self, media: CapturedMedia, object_key: str, *, overwrite: bool = False
    ) -> StorageReceipt:
        if self.fail_upload:
            raise ComponentUnavailable("injected storage upload failure")
        if object_key in self.objects:
            content, receipt = self.objects[object_key]
            if receipt.asset_id == media.asset.asset_id:
                return receipt
            if not overwrite:
                raise FileExistsError(object_key)
        if media.content is not None:
            content = media.content
        else:
            assert media.local_path is not None
            content = media.local_path.read_bytes()
        receipt = StorageReceipt(
            asset_id=media.asset.asset_id,
            object_key=object_key,
            stored_at_utc=self.clock.utc_now(),
            checksum_sha256=hashlib.sha256(content).hexdigest(),
        )
        self.objects[object_key] = (content, receipt)
        return receipt

    async def download(self, object_key: str) -> bytes:
        return self.objects[object_key][0]

    async def delete(self, object_key: str) -> None:
        self.objects.pop(object_key, None)


class InMemoryTagRegistry:
    def __init__(self, entries: Iterable[TagRegistryEntry]) -> None:
        self._entries = {entry.tag_id: entry for entry in entries}

    def resolve(self, tag_id: str) -> TagRegistryEntry | None:
        entry = self._entries.get(tag_id)
        return entry if entry is not None and entry.enabled else None

    def entries(self):
        return tuple(self._entries.values())
