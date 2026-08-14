"""BearVision 3 domain contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .time import UtcDatetime


ContractVersion = Literal["2.0"]
Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")]


class ContractModel(BaseModel):
    """Strict base for serialized contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    contract_schema_version: ContractVersion = "2.0"


class Vector3(BaseModel):
    """Three-dimensional vector using units declared by its containing field."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    x: float
    y: float
    z: float


class BoundingBox(BaseModel):
    """Pixel-space bounding box."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    x_px: float = Field(ge=0)
    y_px: float = Field(ge=0)
    width_px: float = Field(gt=0)
    height_px: float = Field(gt=0)


class TagObservation(ContractModel):
    """One decoded BLE observation from a registered or unknown tag."""

    tag_id: Identifier
    observed_at_utc: UtcDatetime
    observed_at_monotonic_s: float = Field(ge=0)
    rssi_dbm: int = Field(ge=-127, le=20)
    acceleration_mps2: Vector3
    battery_voltage_mv: int | None = Field(default=None, ge=0, le=10_000)


class TagRegistryEntry(ContractModel):
    """Authoritative BLE tag-to-rider mapping."""

    tag_id: Identifier
    rider_id: Identifier
    enabled: bool = True


class PersonDetection(ContractModel):
    """A detector result. It may trigger capture but never assigns identity."""

    frame_id: Identifier
    observed_at_monotonic_s: float = Field(ge=0)
    bounding_box: BoundingBox
    confidence: float = Field(ge=0, le=1)


class RiderAssignmentStatus(StrEnum):
    ASSIGNED = "assigned"
    UNASSIGNED = "unassigned"
    AMBIGUOUS = "ambiguous"


class TagAssignmentEvidence(ContractModel):
    """Acceleration and RSSI evidence calculated for one registered BearTag."""

    tag_id: Identifier
    rider_id: Identifier
    observation_count: int = Field(gt=0)
    mean_motion_delta_mps2: float = Field(ge=0)
    median_rssi_dbm: float = Field(ge=-127, le=20)
    motion_score: float = Field(ge=0, le=1)
    rssi_score: float = Field(ge=0, le=1)
    combined_score: float = Field(ge=0, le=1)
    qualifies: bool


class RiderAssignment(ContractModel):
    """Result of BearTag acceleration-plus-RSSI rider assignment."""

    status: RiderAssignmentStatus
    assigned_at_monotonic_s: float = Field(ge=0)
    rider_id: Identifier | None = None
    tag_id: Identifier | None = None
    candidate_tag_ids: tuple[Identifier, ...] = ()
    evidence: tuple[TagAssignmentEvidence, ...] = ()
    reason: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_assignment(self) -> "RiderAssignment":
        if self.status is RiderAssignmentStatus.ASSIGNED:
            if not self.rider_id or not self.tag_id:
                raise ValueError("assigned result requires rider_id and tag_id")
        elif self.rider_id is not None or self.tag_id is not None:
            raise ValueError("unassigned or ambiguous result cannot contain an assigned rider")
        evidence_ids = {item.tag_id for item in self.evidence if item.qualifies}
        if evidence_ids and evidence_ids != set(self.candidate_tag_ids):
            raise ValueError("candidate_tag_ids must match qualifying BearTag evidence")
        return self


class CaptureRequest(ContractModel):
    """Request to preserve a camera clip around a detection."""

    request_id: Identifier
    requested_at_monotonic_s: float = Field(ge=0)
    pre_roll_s: float = Field(ge=0)
    post_roll_s: float = Field(ge=0)
    assignment: RiderAssignment | None = None


class CaptureStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class MediaAsset(ContractModel):
    """Provider-neutral media reference."""

    asset_id: Identifier
    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(min_length=1, max_length=100)
    size_bytes: int = Field(ge=0)
    created_at_utc: UtcDatetime


class CaptureResult(ContractModel):
    """Terminal result of a capture request."""

    request_id: Identifier
    status: CaptureStatus
    completed_at_monotonic_s: float = Field(ge=0)
    media: MediaAsset | None = None
    error_code: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "CaptureResult":
        if self.status is CaptureStatus.COMPLETED and self.media is None:
            raise ValueError("completed capture requires media")
        if self.status is CaptureStatus.FAILED and not self.error_code:
            raise ValueError("failed capture requires error_code")
        return self


class StorageReceipt(ContractModel):
    """Provider-neutral confirmation that a media asset was stored."""

    asset_id: Identifier
    object_key: str = Field(min_length=1, max_length=1024)
    stored_at_utc: UtcDatetime
    checksum_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
