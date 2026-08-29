"""Typed command and read-model module for the Server Control process seam."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic.alias_generators import to_camel

from bearvision.adapters import SystemClock
from bearvision.config import ServerConfig
from bearvision.contracts import CandidateScore, EdgeJobManifest, JobResultManifest, JobVideo
from bearvision.ports import ManagedJobQueue

from .admin import AdminCatalog, AdminMediaService, UserVideoCatalog
from .registry import (
    BearTagAssignment,
    BearTagRecord,
    FileUserRegistry,
    RegistryData,
    UserRecord,
)
from .worker import ServerWorker


class ProcessModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        extra="forbid",
        populate_by_name=True,
    )


class CommandModel(ProcessModel):
    command_schema_version: Literal["1.0"] = "1.0"


class SnapshotCommand(CommandModel):
    command: Literal["snapshot"]


class SummaryCommand(CommandModel):
    command: Literal["summary"]


class RunOnceCommand(CommandModel):
    command: Literal["run-once"]


class ListJobsCommand(CommandModel):
    command: Literal["list-jobs"]
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=24, ge=1, le=100)
    status: str | None = None
    query: str = ""
    user_id: UUID | None = None


class JobDetailCommand(CommandModel):
    command: Literal["job-detail"]
    job_id: str = Field(min_length=1)


class ListUsersCommand(CommandModel):
    command: Literal["list-users"]
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=100)
    query: str = ""


class ListTagsCommand(CommandModel):
    command: Literal["list-tags"]


class MaterializeMediaCommand(CommandModel):
    command: Literal["materialize-media"]
    job_id: str = Field(min_length=1)
    kind: Literal["video", "thumbnail"]


class ListUserVideosCommand(CommandModel):
    command: Literal["list-user-videos"]
    user_email: str = Field(alias="userId", min_length=1)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=100)


class MaterializeUserMediaCommand(CommandModel):
    command: Literal["materialize-user-media"]
    user_email: str = Field(alias="userId", min_length=1)
    job_id: str = Field(min_length=1)
    kind: Literal["video", "thumbnail"]


class CreateUserCommand(CommandModel):
    command: Literal["create-user"]
    email: str = Field(min_length=1)
    display_name: str = Field(min_length=1)


class UpdateUserEmailCommand(CommandModel):
    command: Literal["update-user-email"]
    user_id: UUID
    email: str = Field(min_length=1)


class CreateTagCommand(CommandModel):
    command: Literal["create-tag"]
    id: str = Field(min_length=1)


class AssignmentCommand(CommandModel):
    id: str | None = Field(default=None, min_length=1)
    user_id: UUID
    bear_tag_id: str = Field(min_length=1)
    valid_from: datetime
    valid_to: datetime

    def assignment(self) -> BearTagAssignment:
        return BearTagAssignment(
            id=self.id or f"assignment-{uuid4().hex}",
            userId=self.user_id,
            bearTagId=self.bear_tag_id,
            validFrom=self.valid_from,
            validTo=self.valid_to,
        )


class CreateAssignmentCommand(AssignmentCommand):
    command: Literal["create-assignment"]


class ValidateAssignmentCommand(AssignmentCommand):
    command: Literal["validate-assignment"]


class RequeueCommand(CommandModel):
    command: Literal["requeue"]
    job_id: str = Field(min_length=1)


ServerCommand = Annotated[
    SnapshotCommand
    | SummaryCommand
    | RunOnceCommand
    | ListJobsCommand
    | JobDetailCommand
    | ListUsersCommand
    | ListTagsCommand
    | MaterializeMediaCommand
    | ListUserVideosCommand
    | MaterializeUserMediaCommand
    | CreateUserCommand
    | UpdateUserEmailCommand
    | CreateTagCommand
    | CreateAssignmentCommand
    | ValidateAssignmentCommand
    | RequeueCommand,
    Field(discriminator="command"),
]
COMMAND_ADAPTER: TypeAdapter[ServerCommand] = TypeAdapter(ServerCommand)


class ReadModel(ProcessModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        extra="allow",
        populate_by_name=True,
    )


class QueueCounts(ProcessModel):
    ready: int = 0
    processing: int = 0
    processed: int = 0
    unresolved: int = 0
    failed: int = 0


class WorkerReadModel(ReadModel):
    status: str


class SummaryReadModel(ProcessModel):
    counts: QueueCounts
    attention_count: int
    worker: WorkerReadModel


class SnapshotReadModel(ProcessModel):
    worker: WorkerReadModel
    queue: dict[str, Any]
    registry: RegistryData


class JobReadModel(ReadModel):
    job_id: str
    status: str
    user_id: UUID | None = None
    display_name: str | None = None
    user_email: str | None = None
    capture_started_at: datetime | None = None
    capture_ended_at: datetime | None = None
    created_at: datetime | None = None
    duration_seconds: float | None = None
    video: JobVideo | None = None
    selected_bear_tag_id: str | None = None
    selected_user_id: UUID | None = None
    assignment_id: str | None = None
    candidates: tuple[CandidateScore, ...] = ()
    reason: str | None = None
    error_code: str | None = None
    metadata_errors: tuple[str, ...] = ()
    manifest: EdgeJobManifest | None = None


class JobPageReadModel(ProcessModel):
    items: tuple[JobReadModel, ...]
    page: int
    page_size: int
    total: int
    page_count: int


class AssignmentReadModel(ProcessModel):
    id: str
    user_id: UUID
    bear_tag_id: str
    valid_from: datetime
    valid_to: datetime
    active: bool


class UserReadModel(ProcessModel):
    id: UUID
    email: str
    display_name: str
    assignments: tuple[AssignmentReadModel, ...] = ()
    active_bear_tags: tuple[str, ...] = ()
    processed_video_count: int = 0


class UserPageReadModel(ProcessModel):
    items: tuple[UserReadModel, ...]
    page: int
    page_size: int
    total: int
    page_count: int


class TagReadModel(ProcessModel):
    id: str
    assignments: tuple[AssignmentReadModel, ...] = ()


class TagListReadModel(ProcessModel):
    items: tuple[TagReadModel, ...]


class UserVideosReadModel(ProcessModel):
    user: UserRecord
    items: tuple[JobReadModel, ...]
    page: int
    page_size: int
    total: int
    page_count: int


class MediaReadModel(ProcessModel):
    path: Path
    content_type: str
    size_bytes: int


class AssignmentValidationReadModel(ProcessModel):
    valid: Literal[True] = True
    assignment: BearTagAssignment


class RequeueReadModel(ProcessModel):
    requeued: bool


CommandResult = (
    SummaryReadModel
    | SnapshotReadModel
    | JobReadModel
    | JobPageReadModel
    | UserPageReadModel
    | TagListReadModel
    | UserVideosReadModel
    | MediaReadModel
    | AssignmentValidationReadModel
    | RequeueReadModel
    | UserRecord
    | BearTagRecord
    | BearTagAssignment
    | JobResultManifest
    | None
)


def parse_command(payload: str) -> ServerCommand:
    """Validate the complete versioned command envelope once."""

    return COMMAND_ADAPTER.validate_json(payload)


def _resolve(base: Path, value: Path) -> Path:
    return value if value.is_absolute() else base / value


def worker_status(config_path: Path) -> WorkerReadModel:
    path = config_path.resolve().parents[1] / "temp/server-worker-status.json"
    payload: dict[str, Any] = {"status": "stopped"}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {"status": "unknown"}
    return WorkerReadModel.model_validate(payload)


def write_worker_status(config_path: Path, **values: Any) -> None:
    path = config_path.resolve().parents[1] / "temp/server-worker-status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    payload = {
        "pid": os.getpid(),
        "updatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **values,
    }
    try:
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ServerCommandModule:
    """Validate, dispatch and shape every short-lived Server Control command."""

    def __init__(
        self,
        config_path: Path,
        config: ServerConfig,
        queue: ManagedJobQueue,
        registry: FileUserRegistry,
    ) -> None:
        self.config_path = config_path
        self.config = config
        self.queue = queue
        self.registry = registry
        self.catalog = AdminCatalog(queue, registry)
        self.user_catalog = UserVideoCatalog(queue, registry)
        self.base = config_path.resolve().parents[1]

    async def execute(self, command: ServerCommand) -> CommandResult:
        if isinstance(command, SnapshotCommand):
            return SnapshotReadModel(
                worker=worker_status(self.config_path),
                queue=self.queue.snapshot(),
                registry=self.registry.load(),
            )
        if isinstance(command, SummaryCommand):
            summary = await self.catalog.summary()
            return SummaryReadModel.model_validate(
                {**summary, "worker": worker_status(self.config_path)}
            )
        if isinstance(command, ListJobsCommand):
            return JobPageReadModel.model_validate(
                await self.catalog.list_jobs(
                    page=command.page,
                    page_size=command.page_size,
                    status=command.status,
                    query=command.query,
                    user_id=str(command.user_id) if command.user_id else None,
                )
            )
        if isinstance(command, JobDetailCommand):
            return JobReadModel.model_validate(await self.catalog.get_job(command.job_id))
        if isinstance(command, ListUsersCommand):
            return UserPageReadModel.model_validate(
                await self.catalog.list_users(
                    page=command.page,
                    page_size=command.page_size,
                    query=command.query,
                )
            )
        if isinstance(command, ListTagsCommand):
            return TagListReadModel.model_validate(self.catalog.list_bear_tags())
        if isinstance(command, MaterializeMediaCommand):
            media = AdminMediaService(
                self.queue,
                _resolve(self.base, self.config.scratch_dir) / "admin-media",
            )
            return MediaReadModel.model_validate(
                await media.materialize(command.job_id, command.kind)
            )
        if isinstance(command, ListUserVideosCommand):
            return UserVideosReadModel.model_validate(
                await self.user_catalog.list_videos(
                    command.user_email,
                    page=command.page,
                    page_size=command.page_size,
                )
            )
        if isinstance(command, MaterializeUserMediaCommand):
            media = AdminMediaService(
                self.queue,
                _resolve(self.base, self.config.scratch_dir) / "app-media",
                registry=self.registry,
            )
            return MediaReadModel.model_validate(
                await media.materialize_for_user(
                    command.user_email,
                    command.job_id,
                    command.kind,
                )
            )
        if isinstance(command, CreateUserCommand):
            return self.registry.create_user(command.email, command.display_name)
        if isinstance(command, UpdateUserEmailCommand):
            return self.registry.update_user_email(command.user_id, command.email)
        if isinstance(command, CreateTagCommand):
            return self.registry.create_bear_tag(command.id)
        if isinstance(command, CreateAssignmentCommand):
            return self.registry.create_assignment(command.assignment())
        if isinstance(command, ValidateAssignmentCommand):
            assignment, _ = self.registry.validate_assignment(command.assignment())
            return AssignmentValidationReadModel(assignment=assignment)
        if isinstance(command, RequeueCommand):
            return RequeueReadModel(requeued=await self.queue.requeue(command.job_id))
        if isinstance(command, RunOnceCommand):
            return await ServerWorker(
                self.queue,
                self.registry,
                SystemClock(),
                self.config.assignment,
            ).run_once()
        raise AssertionError(f"unhandled command type: {type(command).__name__}")


def serialize_result(result: CommandResult) -> str:
    payload = result.model_dump(mode="json", by_alias=True) if result is not None else None
    return json.dumps(payload, separators=(",", ":"))
