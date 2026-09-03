"""Durable, disk-backed processing queue for already downloaded camera clips."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from bearvision.contracts import (
    BearTagJobObservation,
    EdgeJobManifest,
    MediaAsset,
    RuntimeEventKind,
    TagObservation,
)
from bearvision.ports import CapturedClip, CapturedMedia, ClipProcessor, Clock, JobQueue

from .job_package import build_edge_job


RawClipJobStatus = Literal[
    "queued", "processing", "packaging", "uploading", "failed", "completed"
]


@dataclass(frozen=True, slots=True)
class RawClipJobContext:
    """Immutable evidence captured at the camera-to-disk queue seam."""

    capture_started_at_utc: datetime
    capture_ended_at_utc: datetime
    observations: tuple[TagObservation, ...]

    def __post_init__(self) -> None:
        if self.capture_started_at_utc.utcoffset() is None:
            raise ValueError("capture start must include a timezone")
        if self.capture_ended_at_utc <= self.capture_started_at_utc:
            raise ValueError("capture end must be later than capture start")


@dataclass(frozen=True, slots=True)
class RawClipJobSummary:
    job_id: str
    request_id: str
    status: RawClipJobStatus
    processing_attempts: int
    queued_at_utc: datetime
    state_changed_at_utc: datetime
    raw_filename: str
    processed_filename: str | None = None
    failure_id: str | None = None
    failed_step: str | None = None
    technical_error: str | None = None
    media_asset_id: str | None = None
    object_key: str | None = None
    checksum_sha256: str | None = None
    uploaded: bool = False


@dataclass(frozen=True, slots=True)
class RawClipQueueSnapshot:
    counts: dict[str, int]
    current_job: str | None
    oldest_queued_at_utc: datetime | None
    jobs: tuple[RawClipJobSummary, ...]


class RawClipPipeline:
    """Own raw-clip persistence and background work behind one small interface."""

    _STATUSES = ("queued", "processing", "failed", "completed")

    def __init__(
        self,
        *,
        capture_dir: str | Path,
        clock: Clock,
        clip_processor: ClipProcessor | None,
        job_queue: JobQueue,
        edge_device_id: str,
        upload_enabled: bool = True,
        event_sink: Callable[[RuntimeEventKind, dict[str, Any], float], None] | None = None,
    ) -> None:
        self.capture_dir = Path(capture_dir).resolve()
        self.queue_dir = self.capture_dir / ".raw-clip-queue"
        self.clock = clock
        self.clip_processor = clip_processor
        self.job_queue = job_queue
        self.edge_device_id = edge_device_id
        self.upload_enabled = upload_enabled
        self.event_sink = event_sink
        self._queue: asyncio.PriorityQueue[tuple[str, str]] = asyncio.PriorityQueue()
        self._worker: asyncio.Task[None] | None = None
        self._accepting = False
        self._current_job: str | None = None

    async def start(self) -> None:
        if self._worker is not None and not self._worker.done():
            return
        self._queue = asyncio.PriorityQueue()
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        for status in self._STATUSES:
            (self.queue_dir / status).mkdir(parents=True, exist_ok=True)
        self._recover_processing_jobs()
        self._repair_status_group_metadata()
        queued_records = sorted(
            (
                json.loads(path.read_text(encoding="utf-8"))
                for path in (self.queue_dir / "queued").glob("*.json")
            ),
            key=lambda item: (item["queued_at_utc"], item["job_id"]),
        )
        for record in queued_records:
            try:
                self._validate_persisted_raw(record)
            except Exception as exc:
                self._fail_queued_validation(record, exc)
            else:
                await self._queue.put((record["queued_at_utc"], record["job_id"]))
        self._accepting = True
        self._worker = asyncio.create_task(self._run_worker())
        self._emit("clip_queue_snapshot", self._snapshot_payload())
        for record in self._records_in("failed"):
            self._emit_failure(record)

    async def submit(
        self,
        captured_clip: CapturedClip,
        context: RawClipJobContext,
    ) -> RawClipJobSummary:
        if not self._accepting:
            raise RuntimeError("raw clip pipeline is not accepting jobs")
        media = captured_clip.media
        if media.local_path is None:
            raise ValueError("raw clip must be backed by a local file")
        raw_path = media.local_path.resolve()
        if raw_path.parent != self.capture_dir or not raw_path.is_file():
            raise ValueError("raw clip must be an existing file directly under capture_dir")
        if raw_path.name != media.asset.filename:
            raise ValueError("raw clip filename does not match its media asset")
        if raw_path.stat().st_size != media.asset.size_bytes:
            raise ValueError("raw clip size does not match its media asset")
        target = self._path("queued", captured_clip.request_id)
        if any(self._path(status, captured_clip.request_id).exists() for status in self._STATUSES):
            return self._summary(self._load_existing(captured_clip.request_id))
        now = self.clock.utc_now()
        record: dict[str, Any] = {
            "raw_clip_job_schema_version": "1.0",
            "job_id": captured_clip.request_id,
            "request_id": captured_clip.request_id,
            "queued_at_utc": now.isoformat(),
            "state_changed_at_utc": now.isoformat(),
            "status": "queued",
            "processing_attempts": 0,
            "raw_filename": raw_path.name,
            "raw_media": media.asset.model_dump(mode="json"),
            "requested_window": self._window_dict(captured_clip.requested_window),
            "actual_window": self._window_dict(captured_clip.actual_window),
            "capture_started_at_utc": context.capture_started_at_utc.isoformat(),
            "capture_ended_at_utc": context.capture_ended_at_utc.isoformat(),
            "observations": [item.model_dump(mode="json") for item in context.observations],
            "processed_filename": None,
            "debug_video_filename": None,
            "tracking_filename": None,
            "trim_offset_s": None,
            "processed_duration_s": None,
            "upload_status": "pending" if self.upload_enabled else "disabled",
            "object_key": None,
            "latest_failure_id": None,
            "failed_step": None,
            "technical_error": None,
            "retry_checkpoint": "processing",
        }
        self._atomic_write(target, record)
        self._emit_job(record)
        await self._queue.put((record["queued_at_utc"], captured_clip.request_id))
        return self._summary(record)

    async def retry(self, failure_id: str) -> None:
        if not self._accepting:
            raise RuntimeError("raw clip pipeline is not accepting jobs")
        for path in (self.queue_dir / "failed").glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("latest_failure_id") != failure_id:
                continue
            record["technical_error"] = None
            record["failed_step"] = None
            self._transition_record(
                record, path, "queued", destination_group="queued"
            )
            await self._queue.put((record["queued_at_utc"], record["job_id"]))
            return
        raise ValueError(f"failure is unknown or no longer retryable: {failure_id}")

    async def wait_until_idle(self) -> None:
        self.raise_if_failed()
        worker = self._worker
        if worker is None:
            await self._queue.join()
            return
        idle = asyncio.create_task(self._queue.join())
        done, _ = await asyncio.wait(
            {idle, worker}, return_when=asyncio.FIRST_COMPLETED
        )
        if worker in done:
            idle.cancel()
            with suppress(asyncio.CancelledError):
                await idle
            self.raise_if_failed()
            raise RuntimeError("raw clip pipeline worker ended unexpectedly")
        await idle

    @property
    def worker_task(self) -> asyncio.Task[None] | None:
        """Task supervised by the live runtime for critical storage failures."""

        return self._worker

    def raise_if_failed(self) -> None:
        worker = self._worker
        if worker is None or not worker.done() or worker.cancelled():
            return
        failure = worker.exception()
        if failure is not None:
            raise failure

    async def stop(self) -> None:
        self._accepting = False
        worker, self._worker = self._worker, None
        if worker is not None and not worker.done():
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        self._current_job = None

    def snapshot(self) -> RawClipQueueSnapshot:
        records = self._all_records()
        counts = {status: 0 for status in self._STATUSES}
        for record in records:
            status = str(record["status"])
            counts["processing" if status in {"packaging", "uploading"} else status] += 1
        queued = sorted(
            (record for record in records if record["status"] == "queued"),
            key=lambda item: (item["queued_at_utc"], item["job_id"]),
        )
        recent = sorted(
            records,
            key=lambda item: (item["state_changed_at_utc"], item["job_id"]),
            reverse=True,
        )[:20]
        return RawClipQueueSnapshot(
            counts=counts,
            current_job=self._current_job,
            oldest_queued_at_utc=(
                datetime.fromisoformat(queued[0]["queued_at_utc"]) if queued else None
            ),
            jobs=tuple(self._summary(record) for record in recent),
        )

    async def _run_worker(self) -> None:
        while True:
            _, job_id = await self._queue.get()
            self._current_job = job_id
            try:
                try:
                    await self._process_job(job_id)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._fail_job(job_id, exc)
            finally:
                self._current_job = None
                self._queue.task_done()

    async def _process_job(self, job_id: str) -> None:
        record = self._load_existing(job_id)
        current_path = self._path("queued", job_id)
        if record.get("retry_checkpoint") == "uploading":
            await self._resume_upload(record, current_path)
            return
        record["processing_attempts"] += 1
        current_path = self._transition_record(
            record, current_path, "processing", destination_group="processing"
        )
        raw_media = CapturedMedia(
            asset=MediaAsset.model_validate(record["raw_media"]),
            local_path=self.capture_dir / record["raw_filename"],
        )
        media = raw_media
        job_start_s = float(record["actual_window"]["start_monotonic_s"])
        job_end_s = float(record["actual_window"]["end_monotonic_s"])
        capture_started_at = datetime.fromisoformat(record["capture_started_at_utc"])
        capture_ended_at = datetime.fromisoformat(record["capture_ended_at_utc"])
        if self.clip_processor is not None:
            prepared = await self.clip_processor.process(raw_media)
            media = prepared.media
            job_start_s += prepared.source_start_offset_s
            job_end_s = job_start_s + prepared.duration_s
            capture_started_at += timedelta(seconds=prepared.source_start_offset_s)
            capture_ended_at = capture_started_at + timedelta(seconds=prepared.duration_s)
            record["processed_filename"] = media.asset.filename
            record["processed_media"] = media.asset.model_dump(mode="json")
            record["trim_offset_s"] = prepared.source_start_offset_s
            record["processed_duration_s"] = prepared.duration_s
            for event in prepared.trace_events:
                if event.kind == "virtual_cameraman_completed":
                    record["debug_video_filename"] = event.payload.get(
                        "debug_video_filename"
                    )
                    record["tracking_filename"] = event.payload.get("tracking_filename")
                self._emit(
                    event.kind,
                    event.payload,
                    (
                        float(record["actual_window"]["start_monotonic_s"])
                        + event.source_offset_s
                        if event.source_offset_s is not None
                        else self.clock.monotonic()
                    ),
                )
        current_path = self._transition_record(
            record, current_path, "packaging", destination_group="processing"
        )
        source_observations = tuple(
            TagObservation.model_validate(item) for item in record["observations"]
        )
        observations_in_job = tuple(
            item
            for item in source_observations
            if job_start_s <= item.observed_at_monotonic_s <= job_end_s
        )
        manifest, job_observations = build_edge_job(
            job_id=record["job_id"],
            edge_device_id=self.edge_device_id,
            created_at=max(self.clock.utc_now(), capture_ended_at),
            capture_started_at=capture_started_at,
            capture_ended_at=capture_ended_at,
            clip_start_monotonic_s=job_start_s,
            video=media,
            observations=observations_in_job,
        )
        record["manifest"] = manifest.model_dump(mode="json", by_alias=True)
        record["job_observations"] = [
            item.model_dump(mode="json", by_alias=True) for item in job_observations
        ]
        if self.upload_enabled:
            record["upload_status"] = "uploading"
            record["object_key"] = f"input-queue/ready/{record['job_id']}"
            record["retry_checkpoint"] = "uploading"
            current_path = self._transition_record(
                record, current_path, "uploading", destination_group="processing"
            )
            await self.job_queue.publish(manifest, media, job_observations)
            record["upload_status"] = "completed"
            self._emit(
                "clip_uploaded",
                {
                    "asset_id": media.asset.asset_id,
                    "object_key": record["object_key"],
                    "operation_id": f"{record['job_id']}:publish",
                },
                self.clock.monotonic(),
            )
        self._transition_record(
            record, current_path, "completed", destination_group="completed"
        )
        self._emit_resolution(record)

    async def _resume_upload(self, record: dict[str, Any], current_path: Path) -> None:
        manifest = EdgeJobManifest.model_validate(record["manifest"])
        job_observations = tuple(
            BearTagJobObservation.model_validate(item)
            for item in record["job_observations"]
        )
        processed = record.get("processed_media")
        asset = (
            MediaAsset.model_validate(processed)
            if processed is not None
            else MediaAsset.model_validate(record["raw_media"])
        )
        media = CapturedMedia(
            asset=asset,
            local_path=self.capture_dir / asset.filename,
        )
        record["upload_status"] = "uploading"
        current_path = self._transition_record(
            record, current_path, "uploading", destination_group="processing"
        )
        await self.job_queue.publish(manifest, media, job_observations)
        record["upload_status"] = "completed"
        self._emit(
            "clip_uploaded",
            {
                "asset_id": media.asset.asset_id,
                "object_key": record["object_key"],
                "operation_id": f"{record['job_id']}:publish",
            },
            self.clock.monotonic(),
        )
        self._transition_record(
            record, current_path, "completed", destination_group="completed"
        )
        self._emit_resolution(record)

    def _path(self, status: str, job_id: str) -> Path:
        return self.queue_dir / status / f"{job_id}.json"

    def _load_existing(self, job_id: str) -> dict[str, Any]:
        for status in self._STATUSES:
            path = self._path(status, job_id)
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        raise FileNotFoundError(job_id)

    def _all_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for status in self._STATUSES:
            records.extend(self._records_in(status))
        return records

    def _records_in(self, status: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in (self.queue_dir / status).glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            if status != "processing":
                record["status"] = status
            records.append(record)
        return records

    def _transition_record(
        self,
        record: dict[str, Any],
        current_path: Path,
        status: RawClipJobStatus,
        *,
        destination_group: str,
    ) -> Path:
        destination = self._path(destination_group, record["job_id"])
        if destination != current_path:
            os.replace(current_path, destination)
        record["status"] = status
        record["state_changed_at_utc"] = self.clock.utc_now().isoformat()
        self._atomic_write(destination, record)
        self._emit_job(record)
        return destination

    def _fail_job(self, job_id: str, error: Exception) -> None:
        record = self._load_existing(job_id)
        failed_step = str(record["status"])
        checkpoint = "uploading" if failed_step == "uploading" else "processing"
        failure_id = record.get("latest_failure_id") or (
            f"failure-{job_id}-{failed_step}-{record['processing_attempts']}"
        )
        record["latest_failure_id"] = failure_id
        record["failed_step"] = failed_step
        record["technical_error"] = str(error)
        record["retry_checkpoint"] = checkpoint
        current_group = "queued" if failed_step == "queued" else "processing"
        self._transition_record(
            record,
            self._path(current_group, job_id),
            "failed",
            destination_group="failed",
        )
        self._emit_failure(record)

    def _recover_processing_jobs(self) -> None:
        for current_path in sorted((self.queue_dir / "processing").glob("*.json")):
            record = json.loads(current_path.read_text(encoding="utf-8"))
            destination = self._path("queued", record["job_id"])
            os.replace(current_path, destination)
            record["status"] = "queued"
            record["state_changed_at_utc"] = self.clock.utc_now().isoformat()
            self._atomic_write(destination, record)

    def _repair_status_group_metadata(self) -> None:
        for group in ("queued", "failed", "completed"):
            for path in sorted((self.queue_dir / group).glob("*.json")):
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("status") == group:
                    continue
                previous_status = str(record.get("status") or "processing")
                record["status"] = group
                if group == "failed":
                    attempts = int(record.get("processing_attempts", 0))
                    record["failed_step"] = previous_status
                    record["latest_failure_id"] = (
                        f"failure-{record['job_id']}-{previous_status}-{attempts}"
                    )
                    record["technical_error"] = (
                        "Runtime stopped while persisting clip job failure metadata."
                    )
                    record["retry_checkpoint"] = (
                        "uploading" if previous_status == "uploading" else "processing"
                    )
                elif group == "completed":
                    record["upload_status"] = (
                        "completed" if self.upload_enabled else "disabled"
                    )
                self._atomic_write(path, record)

    def _validate_persisted_raw(self, record: dict[str, Any]) -> None:
        if record.get("raw_clip_job_schema_version") != "1.0":
            raise ValueError("unsupported raw clip job schema version")
        raw_path = self.capture_dir / str(record["raw_filename"])
        if raw_path.parent.resolve() != self.capture_dir or not raw_path.is_file():
            raise FileNotFoundError(f"raw clip is missing: {raw_path.name}")
        asset = MediaAsset.model_validate(record["raw_media"])
        if raw_path.stat().st_size != asset.size_bytes:
            raise ValueError("raw clip size does not match persisted media metadata")

    def _fail_queued_validation(
        self, record: dict[str, Any], error: Exception
    ) -> None:
        record["status"] = "queued"
        record["failed_step"] = "validation"
        record["technical_error"] = str(error)
        record["retry_checkpoint"] = "processing"
        record["latest_failure_id"] = f"failure-{record['job_id']}-validation"
        self._transition_record(
            record,
            self._path("queued", record["job_id"]),
            "failed",
            destination_group="failed",
        )

    def _emit_failure(self, record: dict[str, Any]) -> None:
        failed_step = str(record.get("failed_step") or "failed")
        validation_failure = failed_step == "validation"
        self._emit(
            "component_failed",
            {
                "failure_id": record["latest_failure_id"],
                "operation_id": f"{record['job_id']}:{failed_step}",
                "stage": (
                    "post_processing" if failed_step == "processing" else failed_step
                ),
                "component": (
                    "raw_clip_storage"
                    if validation_failure
                    else "job_queue"
                    if failed_step == "uploading"
                    else "clip_processor"
                    if failed_step == "processing"
                    else "job_package"
                ),
                "error": record.get("technical_error") or "Unknown clip job failure",
                "operator_message": (
                    "The raw clip is missing or invalid."
                    if validation_failure
                    else "The clip job could not be completed."
                ),
                "corrective_action": (
                    "Restore the raw clip in capture storage, then retry."
                    if validation_failure
                    else "Review local media and dependencies, then retry the job."
                ),
                "severity": "blocking",
                "retryable": True,
                "scope": "clip_job",
                "job_id": record["job_id"],
            },
            self.clock.monotonic(),
        )

    def _emit_resolution(self, record: dict[str, Any]) -> None:
        failure_id = record.get("latest_failure_id")
        if failure_id:
            self._emit(
                "failure_resolved",
                {
                    "failure_id": failure_id,
                    "operation_id": f"{record['job_id']}:completed",
                },
                self.clock.monotonic(),
            )

    def _emit_job(self, record: dict[str, Any]) -> None:
        if self.event_sink is None:
            return
        summary = self._summary(record)
        snapshot = self.snapshot()
        current_job = snapshot.current_job
        if (
            summary.job_id == current_job
            and summary.status not in {"processing", "packaging", "uploading"}
        ):
            current_job = None
        self._emit(
            "clip_job_updated",
            {
                **self._summary_payload(summary),
                "counts": snapshot.counts,
                "current_job": current_job,
                "oldest_queued_at_utc": (
                    snapshot.oldest_queued_at_utc.isoformat()
                    if snapshot.oldest_queued_at_utc is not None
                    else None
                ),
            },
            self.clock.monotonic(),
        )

    def _snapshot_payload(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        return {
            "counts": snapshot.counts,
            "current_job": snapshot.current_job,
            "oldest_queued_at_utc": (
                snapshot.oldest_queued_at_utc.isoformat()
                if snapshot.oldest_queued_at_utc is not None
                else None
            ),
            "jobs": [self._summary_payload(job) for job in snapshot.jobs],
        }

    @staticmethod
    def _summary_payload(summary: RawClipJobSummary) -> dict[str, Any]:
        return {
            "job_id": summary.job_id,
            "request_id": summary.request_id,
            "status": summary.status,
            "processing_attempts": summary.processing_attempts,
            "queued_at_utc": summary.queued_at_utc.isoformat(),
            "state_changed_at_utc": summary.state_changed_at_utc.isoformat(),
            "raw_filename": summary.raw_filename,
            "processed_filename": summary.processed_filename,
            "failure_id": summary.failure_id,
            "failed_step": summary.failed_step,
            "technical_error": summary.technical_error,
        }

    def _emit(
        self,
        kind: RuntimeEventKind,
        payload: dict[str, Any],
        at_monotonic_s: float | None = None,
    ) -> None:
        if self.event_sink is not None:
            self.event_sink(
                kind,
                payload,
                self.clock.monotonic() if at_monotonic_s is None else at_monotonic_s,
            )

    @staticmethod
    def _window_dict(window: Any) -> dict[str, Any]:
        return {
            "start_monotonic_s": window.start_monotonic_s,
            "end_monotonic_s": window.end_monotonic_s,
            "precision": window.precision.value,
            "basis": window.basis.value,
        }

    @staticmethod
    def _summary(record: dict[str, Any]) -> RawClipJobSummary:
        manifest = record.get("manifest") or {}
        video = manifest.get("video") or {}
        processed_media = record.get("processed_media") or record.get("raw_media") or {}
        return RawClipJobSummary(
            job_id=record["job_id"],
            request_id=record["request_id"],
            status=record["status"],
            processing_attempts=record["processing_attempts"],
            queued_at_utc=datetime.fromisoformat(record["queued_at_utc"]),
            state_changed_at_utc=datetime.fromisoformat(record["state_changed_at_utc"]),
            raw_filename=record["raw_filename"],
            processed_filename=record.get("processed_filename"),
            failure_id=record.get("latest_failure_id"),
            failed_step=record.get("failed_step"),
            technical_error=record.get("technical_error"),
            media_asset_id=processed_media.get("asset_id"),
            object_key=record.get("object_key"),
            checksum_sha256=video.get("sha256"),
            uploaded=record.get("upload_status") == "completed",
        )

    @staticmethod
    def _atomic_write(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
