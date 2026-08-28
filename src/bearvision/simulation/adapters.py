"""Deterministic component adapters for behavioural simulations."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from uuid import UUID

from bearvision.contracts import (
    BearTagJobObservation,
    CaptureRequest,
    EdgeJobManifest,
    JobResultManifest,
    MediaAsset,
    PersonDetection,
    StorageReceipt,
    TagObservation,
    TagRegistryEntry,
)
from bearvision.ports import (
    CapturedClip,
    CapturedMedia,
    CaptureWindow,
    CaptureWindowBasis,
    CaptureWindowPrecision,
    ComponentUnavailable,
    VideoFrame,
    requested_capture_window,
)
from bearvision.queueing import (
    job_package_files,
    normalize_queue_snapshot,
    serialize_result,
    validate_result_destination,
)


class VirtualClock:
    def __init__(self, start_utc: datetime | None = None) -> None:
        self.start_utc = start_utc or datetime(2026, 1, 1, tzinfo=timezone.utc)
        if self.start_utc.utcoffset() != timedelta(0):
            raise ValueError("start_utc must be UTC")
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
        self.available_since_monotonic_s: float | None = None
        self.captures: dict[str, CapturedClip] = {}

    async def connect(self) -> None:
        self.connected = True
        self.available_since_monotonic_s = self.clock.monotonic()

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

    async def capture(self, request: CaptureRequest) -> CapturedClip:
        if not self.connected:
            raise ComponentUnavailable("simulated camera is disconnected")
        if self.fail_capture:
            raise ComponentUnavailable("injected camera capture failure")
        if request.request_id not in self.captures:
            assert self.available_since_monotonic_s is not None
            requested_window = requested_capture_window(
                request,
                earliest_available_monotonic_s=self.available_since_monotonic_s,
            )
            await self.clock.sleep(request.post_roll_s)
            content = f"bearvision-simulated-clip:{request.request_id}".encode()
            self.captures[request.request_id] = CapturedClip(
                request_id=request.request_id,
                media=CapturedMedia(
                    asset=MediaAsset(
                        asset_id=f"asset-{request.request_id}",
                        filename=f"{request.request_id}.mp4",
                        content_type="video/mp4",
                        size_bytes=len(content),
                        created_at_utc=self.clock.utc_now(),
                    ),
                    content=content,
                ),
                requested_window=requested_window,
                actual_window=CaptureWindow(
                    start_monotonic_s=requested_window.start_monotonic_s,
                    end_monotonic_s=requested_window.end_monotonic_s,
                    precision=CaptureWindowPrecision.EXACT,
                    basis=CaptureWindowBasis.SIMULATED_MEDIA_TIMELINE,
                ),
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


class InMemoryJobQueue:
    """Durable-state semantics without Box or filesystem I/O."""

    def __init__(self, *, fail_publish: bool = False) -> None:
        self.fail_publish = fail_publish
        self.packages: dict[str, dict[str, bytes]] = {}
        self.states: dict[str, str] = {}
        self.results: dict[str, JobResultManifest] = {}
        self.processed_users: dict[str, UUID] = {}

    async def publish(
        self,
        manifest: EdgeJobManifest,
        video: CapturedMedia,
        observations: tuple[BearTagJobObservation, ...],
    ) -> bool:
        if self.fail_publish:
            raise ComponentUnavailable("injected job queue publish failure")
        if manifest.job_id in self.states:
            return False
        self.packages[manifest.job_id] = dict(
            job_package_files(manifest, video, observations)
        )
        self.states[manifest.job_id] = "ready"
        return True

    async def acquire_next(self) -> str | None:
        processing = sorted(
            job_id for job_id, state in self.states.items() if state == "processing"
        )
        if processing:
            return processing[0]
        ready = sorted(
            job_id
            for job_id, state in self.states.items()
            if state == "ready" and "READY" in self.packages.get(job_id, {})
        )
        if not ready:
            return None
        self.states[ready[0]] = "processing"
        return ready[0]

    async def read(self, job_id: str, filename: str) -> bytes:
        return self.packages[job_id][filename]

    async def finish(
        self, job_id: str, result: JobResultManifest, user_id: UUID | None = None
    ) -> None:
        validate_result_destination(result, user_id)
        self.results[job_id] = result
        self.packages[job_id]["result.json"] = serialize_result(result)
        self.states[job_id] = result.status
        if result.status == "processed":
            assert user_id is not None
            self.processed_users[job_id] = user_id

    async def requeue(self, job_id: str) -> bool:
        if self.states.get(job_id) not in {"failed", "unresolved"}:
            return False
        self.states[job_id] = "ready"
        self.results.pop(job_id, None)
        self.processed_users.pop(job_id, None)
        self.packages[job_id].pop("result.json", None)
        return True

    def snapshot(self) -> dict:
        jobs = []
        for job_id, status in self.states.items():
            item = {"jobId": job_id, "status": status}
            result = self.results.get(job_id)
            if result is not None:
                item.update(result.model_dump(mode="json", by_alias=True))
            user_id = self.processed_users.get(job_id)
            if user_id is not None:
                item["userId"] = str(user_id)
            jobs.append(item)
        return normalize_queue_snapshot(jobs)


class InMemoryTagRegistry:
    def __init__(self, entries: Iterable[TagRegistryEntry]) -> None:
        self._entries = {entry.tag_id: entry for entry in entries}

    def resolve(self, tag_id: str) -> TagRegistryEntry | None:
        entry = self._entries.get(tag_id)
        return entry if entry is not None and entry.enabled else None

    def entries(self):
        return tuple(self._entries.values())
