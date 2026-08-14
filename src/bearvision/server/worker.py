"""One-at-a-time server worker for complete BearVision cloud jobs."""

from __future__ import annotations

from datetime import timezone
import hashlib
import json

from pydantic import ValidationError

from bearvision.config import AssignmentConfig
from bearvision.contracts import (
    BearTagJobObservation,
    EdgeJobManifest,
    JobResultManifest,
    TagObservation,
)
from bearvision.domain import ALGORITHM_VERSION, TagSelectionStatus, select_bear_tag
from bearvision.ports import Clock, ComponentTimeout, ComponentUnavailable, JobQueue

from typing import Protocol

from .registry import BearTagAssignment, RegistryData


class UserRegistryReader(Protocol):
    def load(self) -> RegistryData: ...

    def resolve_clip(self, tag_id: str, started_at, ended_at) -> BearTagAssignment | None: ...

    def intersects_assignment(self, tag_id: str, started_at, ended_at) -> bool: ...


class InvalidJob(ValueError):
    """A permanent problem in an Edge job package."""


class ServerWorker:
    def __init__(
        self,
        queue: JobQueue,
        registry: UserRegistryReader,
        clock: Clock,
        assignment_policy: AssignmentConfig | None = None,
    ) -> None:
        self.queue = queue
        self.registry = registry
        self.clock = clock
        self.assignment_policy = assignment_policy or AssignmentConfig()

    async def run_once(self) -> JobResultManifest | None:
        job_id = await self.queue.acquire_next()
        if job_id is None:
            return None
        try:
            manifest, video, observations = await self._load(job_id)
            result = self._decide(manifest, observations)
            await self.queue.finish(job_id, result, result.selected_user_email)
            return result
        except (ComponentTimeout, ComponentUnavailable):
            # Keep the claimed processing folder durable; the polling loop retries it.
            raise
        except (InvalidJob, ValidationError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            result = self._failure(job_id, "INVALID_JOB", str(exc))
            await self.queue.finish(job_id, result)
            return result
        except Exception as exc:
            result = self._failure(job_id, "TECHNICAL_ERROR", str(exc))
            await self.queue.finish(job_id, result)
            return result

    async def _load(
        self, job_id: str
    ) -> tuple[EdgeJobManifest, bytes, tuple[BearTagJobObservation, ...]]:
        manifest = EdgeJobManifest.model_validate_json(await self.queue.read(job_id, "manifest.json"))
        if manifest.job_id != job_id:
            raise InvalidJob("manifest jobId does not match its queue folder")
        video = await self.queue.read(job_id, manifest.video.filename)
        if len(video) != manifest.video.size_bytes:
            raise InvalidJob("video size does not match manifest")
        if hashlib.sha256(video).hexdigest() != manifest.video.sha256:
            raise InvalidJob("video SHA-256 does not match manifest")
        raw_observations = await self.queue.read(job_id, manifest.observations_filename)
        observations: list[BearTagJobObservation] = []
        for line_number, line in enumerate(raw_observations.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = BearTagJobObservation.model_validate_json(line)
            except ValidationError as exc:
                raise InvalidJob(f"invalid observation at line {line_number}: {exc}") from exc
            if item.offset_ms > manifest.duration.total_seconds() * 1000:
                raise InvalidJob(f"observation at line {line_number} is outside the clip")
            observations.append(item)
        return manifest, video, tuple(observations)

    def _decide(
        self,
        manifest: EdgeJobManifest,
        job_observations: tuple[BearTagJobObservation, ...],
    ) -> JobResultManifest:
        registry = self.registry.load()
        observations = tuple(
            TagObservation(
                tag_id=item.bear_tag_id,
                observed_at_utc=item.observed_at(manifest),
                observed_at_monotonic_s=item.offset_ms / 1000,
                rssi_dbm=item.rssi_dbm,
                acceleration_mps2=item.acceleration_mps2,
            )
            for item in job_observations
        )
        selection = select_bear_tag(
            observations,
            (item.id for item in registry.bear_tags),
            clip_start_monotonic_s=0,
            clip_end_monotonic_s=manifest.duration.total_seconds(),
            **self.assignment_policy.model_dump(),
        )
        processed_at = self.clock.utc_now().astimezone(timezone.utc)
        if selection.status is TagSelectionStatus.UNASSIGNED:
            return JobResultManifest(
                jobId=manifest.job_id,
                status="unresolved",
                processedAt=processed_at,
                algorithmVersion=ALGORITHM_VERSION,
                candidates=selection.evidence,
                reason=selection.reason,
                errorCode="NO_QUALIFIED_BEARTAG",
            )
        if selection.status is TagSelectionStatus.AMBIGUOUS:
            return JobResultManifest(
                jobId=manifest.job_id,
                status="unresolved",
                processedAt=processed_at,
                algorithmVersion=ALGORITHM_VERSION,
                candidates=selection.evidence,
                reason=selection.reason,
                errorCode="AMBIGUOUS_BEARTAG",
            )
        assert selection.selected_tag_id is not None
        assignment = self.registry.resolve_clip(
            selection.selected_tag_id,
            manifest.capture_started_at,
            manifest.capture_ended_at,
        )
        if assignment is None:
            intersects = self.registry.intersects_assignment(
                selection.selected_tag_id,
                manifest.capture_started_at,
                manifest.capture_ended_at,
            )
            return JobResultManifest(
                jobId=manifest.job_id,
                status="unresolved",
                processedAt=processed_at,
                algorithmVersion=ALGORITHM_VERSION,
                selectedBearTagId=selection.selected_tag_id,
                candidates=selection.evidence,
                reason=(
                    "clip crosses a BearTag assignment boundary"
                    if intersects
                    else "selected BearTag has no assignment covering the complete clip"
                ),
                errorCode="ASSIGNMENT_BOUNDARY" if intersects else "NO_VALID_ASSIGNMENT",
            )
        return JobResultManifest(
            jobId=manifest.job_id,
            status="processed",
            processedAt=processed_at,
            algorithmVersion=ALGORITHM_VERSION,
            selectedBearTagId=selection.selected_tag_id,
            selectedUserEmail=assignment.user_id,
            assignmentId=assignment.id,
            candidates=selection.evidence,
            reason=f"{selection.reason}; one assignment covers the complete clip",
        )

    def _failure(self, job_id: str, code: str, reason: str) -> JobResultManifest:
        return JobResultManifest(
            jobId=job_id,
            status="failed",
            processedAt=self.clock.utc_now().astimezone(timezone.utc),
            algorithmVersion=ALGORITHM_VERSION,
            reason=reason or code,
            errorCode=code,
        )
