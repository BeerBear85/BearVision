"""Persistent user and historical BearTag assignment registry."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


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
    id: str
    display_name: str = Field(alias="displayName", min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_id(self) -> "UserRecord":
        if self.id != normalize_user_email(self.id):
            raise ValueError("user id must be a trimmed lowercase email address")
        return self


class BearTagRecord(RegistryModel):
    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")


class BearTagAssignment(RegistryModel):
    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    user_id: str = Field(alias="userId")
    bear_tag_id: str = Field(alias="bearTagId")
    valid_from: AwareDatetime = Field(alias="validFrom")
    valid_to: AwareDatetime = Field(alias="validTo")

    @model_validator(mode="after")
    def validate_interval(self) -> "BearTagAssignment":
        if self.user_id != normalize_user_email(self.user_id):
            raise ValueError("assignment userId must be normalized")
        if self.valid_to <= self.valid_from:
            raise ValueError("validTo must be later than validFrom")
        return self


class RegistryData(RegistryModel):
    schema_version: Literal[1] = Field(alias="schemaVersion", default=1)
    users: tuple[UserRecord, ...] = ()
    bear_tags: tuple[BearTagRecord, ...] = Field(alias="bearTags", default=())
    assignments: tuple[BearTagAssignment, ...] = ()

    @model_validator(mode="after")
    def validate_graph(self) -> "RegistryData":
        user_ids = [item.id for item in self.users]
        tag_ids = [item.id for item in self.bear_tags]
        assignment_ids = [item.id for item in self.assignments]
        if len(user_ids) != len(set(user_ids)):
            raise ValueError("user ids must be unique")
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


class FileUserRegistry:
    """Small JSON registry with validation and atomic replacement writes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> RegistryData:
        if not self.path.exists():
            return RegistryData()
        return RegistryData.model_validate_json(self.path.read_text(encoding="utf-8"))

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
        item = UserRecord(id=normalize_user_email(email), displayName=display_name.strip())
        updated = data.model_copy(update={"users": (*data.users, item)})
        updated = RegistryData.model_validate(updated.model_dump())
        self._save(updated)
        return item

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
        normalized = assignment.model_copy(
            update={"user_id": normalize_user_email(assignment.user_id)}
        )
        updated = data.model_copy(update={"assignments": (*data.assignments, normalized)})
        updated = RegistryData.model_validate(updated.model_dump())
        return normalized, updated

    def resolve_clip(self, tag_id: str, started_at, ended_at) -> BearTagAssignment | None:
        """Return the sole assignment covering the complete half-open clip interval."""

        matches = [
            item
            for item in self.load().assignments
            if item.bear_tag_id == tag_id
            and item.valid_from <= started_at
            and ended_at <= item.valid_to
        ]
        return matches[0] if len(matches) == 1 else None

    def intersects_assignment(self, tag_id: str, started_at, ended_at) -> bool:
        return any(
            item.bear_tag_id == tag_id
            and item.valid_from < ended_at
            and started_at < item.valid_to
            for item in self.load().assignments
        )


class InMemoryUserRegistry:
    """Validated registry adapter for deterministic server simulations."""

    def __init__(self, data: RegistryData) -> None:
        self.data = RegistryData.model_validate(data.model_dump())

    def load(self) -> RegistryData:
        return self.data

    def resolve_clip(self, tag_id: str, started_at, ended_at) -> BearTagAssignment | None:
        matches = [
            item
            for item in self.data.assignments
            if item.bear_tag_id == tag_id
            and item.valid_from <= started_at
            and ended_at <= item.valid_to
        ]
        return matches[0] if len(matches) == 1 else None

    def intersects_assignment(self, tag_id: str, started_at, ended_at) -> bool:
        return any(
            item.bear_tag_id == tag_id
            and item.valid_from < ended_at
            and started_at < item.valid_to
            for item in self.data.assignments
        )
