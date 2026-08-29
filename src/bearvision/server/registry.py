"""Persistent user and historical BearTag assignment registry."""

from __future__ import annotations

from datetime import datetime
import os
import json
from pathlib import Path
import re
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bearvision.config import AssignmentConfig
from bearvision.contracts import (
    BearTagJobObservation,
    EdgeJobManifest,
    JobResultManifest,
    TagObservation,
)
from bearvision.contracts.time import UtcDatetime
from bearvision.domain import ALGORITHM_VERSION, TagSelectionStatus, select_bear_tag


EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def normalize_user_email(value: str) -> str:
    """Normalize only case and surrounding whitespace; aliases remain significant."""

    normalized = value.strip().lower()
    if not EMAIL_PATTERN.fullmatch(normalized):
        raise ValueError("user id must be an email address")
    return normalized


class RegistryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class UserRecord(RegistryModel):
    id: UUID = Field(default_factory=uuid4)
    email: str
    display_name: str = Field(alias="displayName", min_length=1, max_length=200)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = normalize_user_email(value)
        if value != normalized:
            raise ValueError("email must be trimmed and lowercase")
        return normalized


class BearTagRecord(RegistryModel):
    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")


class BearTagAssignment(RegistryModel):
    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    user_id: UUID = Field(alias="userId")
    bear_tag_id: str = Field(alias="bearTagId")
    valid_from: UtcDatetime = Field(alias="validFrom")
    valid_to: UtcDatetime = Field(alias="validTo")

    @model_validator(mode="after")
    def validate_interval(self) -> "BearTagAssignment":
        if self.valid_to <= self.valid_from:
            raise ValueError("validTo must be later than validFrom")
        return self


class RegistryData(RegistryModel):
    """One immutable identity snapshot used for a complete server decision."""

    schema_version: Literal[2] = Field(alias="schemaVersion", default=2)
    users: tuple[UserRecord, ...] = ()
    bear_tags: tuple[BearTagRecord, ...] = Field(alias="bearTags", default=())
    assignments: tuple[BearTagAssignment, ...] = ()

    @model_validator(mode="after")
    def validate_graph(self) -> "RegistryData":
        user_ids = [item.id for item in self.users]
        user_emails = [item.email for item in self.users]
        tag_ids = [item.id for item in self.bear_tags]
        assignment_ids = [item.id for item in self.assignments]
        if len(user_ids) != len(set(user_ids)):
            raise ValueError("user ids must be unique")
        if len(user_emails) != len(set(user_emails)):
            raise ValueError("user emails must be unique")
        if len(tag_ids) != len(set(tag_ids)):
            raise ValueError("BearTag ids must be unique")
        if len(assignment_ids) != len(set(assignment_ids)):
            raise ValueError("assignment ids must be unique")
        for item in self.assignments:
            if item.user_id not in user_ids:
                raise ValueError(f"unknown assignment user: {item.user_id}")
            if item.bear_tag_id not in tag_ids:
                raise ValueError(f"unknown assignment BearTag: {item.bear_tag_id}")
        ordered = sorted(self.assignments, key=lambda item: (item.bear_tag_id, item.valid_from))
        for previous, current in zip(ordered, ordered[1:]):
            if (
                previous.bear_tag_id == current.bear_tag_id
                and current.valid_from < previous.valid_to
            ):
                raise ValueError(
                    f"overlapping assignments for BearTag {current.bear_tag_id}"
                )
        return self

    def decide_job(
        self,
        manifest: EdgeJobManifest,
        job_observations: tuple[BearTagJobObservation, ...],
        *,
        assignment_policy: AssignmentConfig,
        processed_at: datetime,
    ) -> JobResultManifest:
        """Select and historically resolve identity using only this snapshot."""

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
            (item.id for item in self.bear_tags),
            clip_start_monotonic_s=0,
            clip_end_monotonic_s=manifest.duration.total_seconds(),
            **assignment_policy.model_dump(),
        )
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
        assignment = self._resolve_clip(
            selection.selected_tag_id,
            manifest.capture_started_at,
            manifest.capture_ended_at,
        )
        if assignment is None:
            intersects = self._intersects_assignment(
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
            selectedUserId=assignment.user_id,
            assignmentId=assignment.id,
            candidates=selection.evidence,
            reason=f"{selection.reason}; one assignment covers the complete clip",
        )

    def _resolve_clip(
        self,
        tag_id: str,
        started_at: datetime,
        ended_at: datetime,
    ) -> BearTagAssignment | None:
        matches = [
            item
            for item in self.assignments
            if item.bear_tag_id == tag_id
            and item.valid_from <= started_at
            and ended_at <= item.valid_to
        ]
        return matches[0] if len(matches) == 1 else None

    def _intersects_assignment(
        self,
        tag_id: str,
        started_at: datetime,
        ended_at: datetime,
    ) -> bool:
        return any(
            item.bear_tag_id == tag_id
            and item.valid_from < ended_at
            and started_at < item.valid_to
            for item in self.assignments
        )


class FileUserRegistry:
    """Small JSON registry with validation and atomic replacement writes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> RegistryData:
        if not self.path.exists():
            return RegistryData()
        raw: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        if raw.get("schemaVersion") == 1:
            return self._migrate_v1(raw)
        return RegistryData.model_validate(raw)

    @staticmethod
    def _migrate_v1(raw: dict[str, Any]) -> RegistryData:
        """Read the former email-as-identity registry without losing history."""

        users = raw.get("users", [])
        ids_by_email = {
            normalize_user_email(str(user["id"])): uuid5(
                NAMESPACE_URL, f"bearvision:user:{normalize_user_email(str(user['id']))}"
            )
            for user in users
        }
        migrated = {
            "schemaVersion": 2,
            "users": [
                {
                    "id": str(ids_by_email[normalize_user_email(str(user["id"]))]),
                    "email": normalize_user_email(str(user["id"])),
                    "displayName": user["displayName"],
                }
                for user in users
            ],
            "bearTags": raw.get("bearTags", []),
            "assignments": [
                {
                    **assignment,
                    "userId": str(
                        ids_by_email[normalize_user_email(str(assignment["userId"]))]
                    ),
                }
                for assignment in raw.get("assignments", [])
            ],
        }
        return RegistryData.model_validate(migrated)

    def _save(self, data: RegistryData) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        payload = data.model_dump_json(by_alias=True, indent=2)
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)

    def create_user(self, email: str, display_name: str) -> UserRecord:
        data = self.load()
        item = UserRecord(email=normalize_user_email(email), displayName=display_name.strip())
        updated = data.model_copy(update={"users": (*data.users, item)})
        updated = RegistryData.model_validate(updated.model_dump())
        self._save(updated)
        return item

    def update_user_email(self, user_id: UUID | str, email: str) -> UserRecord:
        data = self.load()
        normalized_id = UUID(str(user_id))
        replacement: UserRecord | None = None
        users: list[UserRecord] = []
        for user in data.users:
            if user.id == normalized_id:
                replacement = user.model_copy(update={"email": normalize_user_email(email)})
                users.append(replacement)
            else:
                users.append(user)
        if replacement is None:
            raise FileNotFoundError("user not found")
        updated = RegistryData.model_validate(
            data.model_copy(update={"users": tuple(users)}).model_dump()
        )
        self._save(updated)
        return replacement

    def find_user_by_email(self, email: str) -> UserRecord | None:
        normalized = normalize_user_email(email)
        return next((user for user in self.load().users if user.email == normalized), None)

    def create_bear_tag(self, tag_id: str) -> BearTagRecord:
        data = self.load()
        item = BearTagRecord(id=tag_id)
        updated = data.model_copy(update={"bear_tags": (*data.bear_tags, item)})
        updated = RegistryData.model_validate(updated.model_dump())
        self._save(updated)
        return item

    def create_assignment(self, assignment: BearTagAssignment) -> BearTagAssignment:
        normalized, updated = self.validate_assignment(assignment)
        self._save(updated)
        return normalized

    def validate_assignment(
        self, assignment: BearTagAssignment
    ) -> tuple[BearTagAssignment, RegistryData]:
        """Validate a proposed assignment without mutating the registry."""

        data = self.load()
        updated = data.model_copy(update={"assignments": (*data.assignments, assignment)})
        updated = RegistryData.model_validate(updated.model_dump())
        return assignment, updated

class InMemoryUserRegistry:
    """Validated registry adapter for deterministic server simulations."""

    def __init__(self, data: RegistryData) -> None:
        self.data = RegistryData.model_validate(data.model_dump())

    def load(self) -> RegistryData:
        return self.data
