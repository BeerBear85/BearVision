"""Versioned cloud job contracts shared by Edge and server."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from .models import Vector3


class JobContractModel(BaseModel):
    """Strict camel-case model used for files exchanged through cloud storage."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class JobVideo(JobContractModel):
    filename: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(alias="mimeType", min_length=1, max_length=100)
    size_bytes: int = Field(alias="sizeBytes", ge=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class EdgeJobManifest(JobContractModel):
    schema_version: Literal[1] = Field(alias="schemaVersion", default=1)
    job_id: str = Field(
        alias="jobId", min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"
    )
    edge_device_id: str = Field(
        alias="edgeDeviceId", min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"
    )
    created_at: AwareDatetime = Field(alias="createdAt")
    capture_started_at: AwareDatetime = Field(alias="captureStartedAt")
    capture_ended_at: AwareDatetime = Field(alias="captureEndedAt")
    video: JobVideo
    observations_filename: Literal["beartag-data.ndjson"] = Field(
        alias="observationsFilename", default="beartag-data.ndjson"
    )

    @model_validator(mode="after")
    def validate_times(self) -> "EdgeJobManifest":
        if self.capture_ended_at <= self.capture_started_at:
            raise ValueError("captureEndedAt must be later than captureStartedAt")
        return self

    @property
    def duration(self) -> timedelta:
        return self.capture_ended_at - self.capture_started_at


class BearTagJobObservation(JobContractModel):
    """One anonymous observation; time is milliseconds from clip start."""

    bear_tag_id: str = Field(
        alias="bearTagId", min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"
    )
    offset_ms: int = Field(alias="offsetMs", ge=0)
    rssi_dbm: int = Field(alias="rssiDbm", ge=-127, le=20)
    acceleration_mps2: Vector3 = Field(alias="accelerationMps2")

    def observed_at(self, manifest: EdgeJobManifest) -> datetime:
        return manifest.capture_started_at + timedelta(milliseconds=self.offset_ms)


class CandidateScore(JobContractModel):
    bear_tag_id: str = Field(alias="bearTagId")
    observation_count: int = Field(alias="observationCount", ge=1)
    mean_motion_delta_mps2: float = Field(alias="meanMotionDeltaMps2", ge=0)
    median_rssi_dbm: float = Field(alias="medianRssiDbm", ge=-127, le=20)
    motion_score: float = Field(alias="motionScore", ge=0, le=1)
    rssi_score: float = Field(alias="rssiScore", ge=0, le=1)
    combined_score: float = Field(alias="combinedScore", ge=0, le=1)
    qualifies: bool


class JobResultManifest(JobContractModel):
    schema_version: Literal[1] = Field(alias="schemaVersion", default=1)
    job_id: str = Field(alias="jobId", min_length=1)
    status: Literal["processed", "unresolved", "failed"]
    processed_at: AwareDatetime = Field(alias="processedAt")
    algorithm_version: str = Field(alias="algorithmVersion", min_length=1)
    selected_bear_tag_id: str | None = Field(alias="selectedBearTagId", default=None)
    selected_user_email: str | None = Field(alias="selectedUserEmail", default=None)
    assignment_id: str | None = Field(alias="assignmentId", default=None)
    candidates: tuple[CandidateScore, ...] = ()
    reason: str = Field(min_length=1, max_length=1000)
    error_code: str | None = Field(alias="errorCode", default=None)
