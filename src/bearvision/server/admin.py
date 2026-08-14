"""Administrative read models and Python-owned media preparation."""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from bearvision.adapters.ffmpeg import FfmpegVideoClipper
from bearvision.contracts import EdgeJobManifest, JobResultManifest
from bearvision.ports import ComponentUnavailable, ManagedJobQueue

from .registry import FileUserRegistry, normalize_user_email


TERMINAL_STATES = frozenset({"processed", "unresolved", "failed"})


def _page(value: int, *, minimum: int = 1) -> int:
    if value < minimum:
        raise ValueError(f"value must be at least {minimum}")
    return value


class AdminCatalog:
    """Build UI-specific views while keeping storage details out of Node.js."""

    def __init__(self, queue: ManagedJobQueue, registry: FileUserRegistry) -> None:
        self.queue = queue
        self.registry = registry

    async def summary(self) -> dict[str, Any]:
        entries = self.queue.admin_list_jobs()
        counts = Counter(item["status"] for item in entries)
        return {
            "counts": {
                status: counts[status]
                for status in ("ready", "processing", "processed", "unresolved", "failed")
            },
            "attentionCount": counts["unresolved"] + counts["failed"],
        }

    async def list_jobs(
        self,
        *,
        page: int = 1,
        page_size: int = 24,
        status: str | None = None,
        query: str = "",
        user_id: str | None = None,
    ) -> dict[str, Any]:
        page = _page(page)
        page_size = _page(page_size)
        if page_size > 100:
            raise ValueError("page size must not exceed 100")
        if status and status not in {
            "ready",
            "processing",
            "processed",
            "unresolved",
            "failed",
        }:
            raise ValueError("invalid job status")

        entries = self.queue.admin_list_jobs()
        if status:
            entries = [item for item in entries if item["status"] == status]
        if user_id:
            normalized = user_id.strip().lower()
            entries = [item for item in entries if item.get("userEmail") == normalized]

        normalized_query = query.strip().lower()
        if normalized_query:
            matched: list[dict[str, str]] = []
            for item in entries:
                simple = " ".join(item.values()).lower()
                if normalized_query in simple:
                    matched.append(item)
                    continue
                details = await self._details(item)
                searchable = " ".join(
                    str(details.get(key, ""))
                    for key in (
                        "selectedBearTagId",
                        "selectedUserEmail",
                        "displayName",
                        "reason",
                    )
                ).lower()
                if normalized_query in searchable:
                    matched.append(item)
            entries = matched

        # Job IDs are stable and deterministic. Details for only the requested
        # page are then loaded from Box, avoiding a manifest download per job.
        entries.sort(key=lambda item: item["jobId"], reverse=True)
        total = len(entries)
        offset = (page - 1) * page_size
        items = [await self._details(item) for item in entries[offset : offset + page_size]]
        return {
            "items": items,
            "page": page,
            "pageSize": page_size,
            "total": total,
            "pageCount": (total + page_size - 1) // page_size,
        }

    async def get_job(self, job_id: str) -> dict[str, Any]:
        entry = next(
            (item for item in self.queue.admin_list_jobs() if item["jobId"] == job_id),
            None,
        )
        if entry is None:
            raise FileNotFoundError(job_id)
        return await self._details(entry, include_manifest=True)

    async def list_users(
        self, *, page: int = 1, page_size: int = 50, query: str = ""
    ) -> dict[str, Any]:
        page = _page(page)
        page_size = _page(page_size)
        if page_size > 100:
            raise ValueError("page size must not exceed 100")
        data = self.registry.load()
        now = datetime.now(timezone.utc)
        video_counts = Counter(
            item.get("userEmail")
            for item in self.queue.admin_list_jobs()
            if item["status"] == "processed"
        )
        assignments_by_user: dict[str, list[dict[str, Any]]] = {}
        for assignment in sorted(data.assignments, key=lambda item: item.valid_from, reverse=True):
            payload = assignment.model_dump(mode="json", by_alias=True)
            payload["active"] = assignment.valid_from <= now < assignment.valid_to
            assignments_by_user.setdefault(assignment.user_id, []).append(payload)

        users = []
        for user in data.users:
            assignments = assignments_by_user.get(user.id, [])
            active_tags = [
                item["bearTagId"] for item in assignments if bool(item["active"])
            ]
            users.append(
                {
                    **user.model_dump(mode="json", by_alias=True),
                    "assignments": assignments,
                    "activeBearTags": active_tags,
                    "processedVideoCount": video_counts[user.id],
                }
            )
        normalized_query = query.strip().lower()
        if normalized_query:
            users = [
                item
                for item in users
                if normalized_query
                in " ".join(
                    (
                        item["id"],
                        item["displayName"],
                        *item["activeBearTags"],
                    )
                ).lower()
            ]
        users.sort(key=lambda item: (item["displayName"].casefold(), item["id"]))
        total = len(users)
        offset = (page - 1) * page_size
        return {
            "items": users[offset : offset + page_size],
            "page": page,
            "pageSize": page_size,
            "total": total,
            "pageCount": (total + page_size - 1) // page_size,
        }

    def list_bear_tags(self) -> dict[str, Any]:
        data = self.registry.load()
        assignments_by_tag: dict[str, list[dict[str, Any]]] = {}
        now = datetime.now(timezone.utc)
        for assignment in data.assignments:
            payload = assignment.model_dump(mode="json", by_alias=True)
            payload["active"] = assignment.valid_from <= now < assignment.valid_to
            assignments_by_tag.setdefault(assignment.bear_tag_id, []).append(payload)
        return {
            "items": [
                {
                    **tag.model_dump(mode="json", by_alias=True),
                    "assignments": assignments_by_tag.get(tag.id, []),
                }
                for tag in data.bear_tags
            ]
        }

    async def _details(
        self, entry: dict[str, str], *, include_manifest: bool = False
    ) -> dict[str, Any]:
        manifest: EdgeJobManifest | None = None
        metadata_errors: list[str] = []
        try:
            manifest = EdgeJobManifest.model_validate_json(
                await self.queue.admin_read(entry["jobId"], "manifest.json")
            )
        except (FileNotFoundError, ValidationError) as exc:
            metadata_errors.append(f"manifest: {exc}")
        result: JobResultManifest | None = None
        if entry["status"] in TERMINAL_STATES:
            try:
                result = JobResultManifest.model_validate_json(
                    await self.queue.admin_read(entry["jobId"], "result.json")
                )
            except (FileNotFoundError, ValidationError) as exc:
                metadata_errors.append(f"result: {exc}")
        users = {item.id: item.display_name for item in self.registry.load().users}
        user_email = (
            result.selected_user_email if result is not None else entry.get("userEmail")
        )
        payload: dict[str, Any] = {**entry, "displayName": users.get(user_email or "")}
        if manifest is not None:
            payload.update(
                {
                    "captureStartedAt": manifest.capture_started_at.isoformat(),
                    "captureEndedAt": manifest.capture_ended_at.isoformat(),
                    "createdAt": manifest.created_at.isoformat(),
                    "durationSeconds": manifest.duration.total_seconds(),
                    "video": manifest.video.model_dump(mode="json", by_alias=True),
                }
            )
        if result is not None:
            payload.update(result.model_dump(mode="json", by_alias=True))
        if metadata_errors:
            payload["metadataErrors"] = metadata_errors
        if include_manifest and manifest is not None:
            payload["manifest"] = manifest.model_dump(mode="json", by_alias=True)
        return payload


class UserVideoCatalog:
    """Minimal read model exposed to the passwordless LAN application."""

    def __init__(self, queue: ManagedJobQueue, registry: FileUserRegistry) -> None:
        self.queue = queue
        self.registry = registry
        self.admin_catalog = AdminCatalog(queue, registry)

    async def list_videos(
        self, user_email: str, *, page: int = 1, page_size: int = 50
    ) -> dict[str, Any]:
        normalized = normalize_user_email(user_email)
        user = next(
            (item for item in self.registry.load().users if item.id == normalized),
            None,
        )
        if user is None:
            raise FileNotFoundError("user not found")

        jobs = await self.admin_catalog.list_jobs(
            page=page,
            page_size=page_size,
            status="processed",
            user_id=normalized,
        )
        items = []
        for item in jobs["items"]:
            public_item = {
                key: item[key]
                for key in (
                    "jobId",
                    "captureStartedAt",
                    "captureEndedAt",
                    "createdAt",
                    "durationSeconds",
                    "video",
                )
                if key in item
            }
            items.append(public_item)
        return {
            "user": {"email": user.id, "displayName": user.display_name},
            "items": items,
            "page": jobs["page"],
            "pageSize": jobs["pageSize"],
            "total": jobs["total"],
            "pageCount": jobs["pageCount"],
        }


class AdminMediaService:
    """Download and verify media, then let Python generate cached thumbnails."""

    def __init__(
        self,
        queue: ManagedJobQueue,
        cache_root: Path,
        *,
        ffmpeg_path: str | Path | None = None,
    ) -> None:
        self.queue = queue
        self.cache_root = cache_root
        self.ffmpeg_path = FfmpegVideoClipper._resolve_executable(
            ffmpeg_path, "BEARVISION_FFMPEG", "ffmpeg"
        )

    async def materialize(self, job_id: str, kind: str) -> dict[str, Any]:
        if kind not in {"video", "thumbnail"}:
            raise ValueError("media kind must be video or thumbnail")
        manifest = EdgeJobManifest.model_validate_json(
            await self.queue.admin_read(job_id, "manifest.json")
        )
        if not manifest.video.mime_type.startswith("video/"):
            raise ValueError("job media is not a video")
        video = await self._materialize_video(job_id, manifest)
        if kind == "video":
            return {
                "path": str(video.resolve()),
                "contentType": manifest.video.mime_type,
                "sizeBytes": video.stat().st_size,
            }
        thumbnail = self.cache_root / job_id / "thumbnail.jpg"
        if not thumbnail.is_file():
            await asyncio.to_thread(self._create_thumbnail, video, thumbnail)
        return {
            "path": str(thumbnail.resolve()),
            "contentType": "image/jpeg",
            "sizeBytes": thumbnail.stat().st_size,
        }

    async def materialize_for_user(
        self, user_email: str, job_id: str, kind: str
    ) -> dict[str, Any]:
        """Materialize media only when the processed job belongs to the user."""

        normalized = normalize_user_email(user_email)
        owns_job = any(
            item.get("status") == "processed"
            and item.get("userEmail") == normalized
            and item.get("jobId") == job_id
            for item in self.queue.admin_list_jobs()
        )
        if not owns_job:
            raise FileNotFoundError("video not found for user")
        return await self.materialize(job_id, kind)

    async def _materialize_video(
        self, job_id: str, manifest: EdgeJobManifest
    ) -> Path:
        suffix = Path(manifest.video.filename).suffix.lower() or ".mp4"
        directory = self.cache_root / job_id
        destination = directory / f"video{suffix}"
        checksum_file = directory / "video.sha256"
        if (
            destination.is_file()
            and destination.stat().st_size == manifest.video.size_bytes
            and checksum_file.is_file()
            and checksum_file.read_text(encoding="ascii").strip() == manifest.video.sha256
        ):
            return destination

        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f".video-{uuid4().hex}{suffix}"
        try:
            await self.queue.admin_download(job_id, manifest.video.filename, temporary)
            digest = await asyncio.to_thread(_sha256, temporary)
            if temporary.stat().st_size != manifest.video.size_bytes:
                raise ValueError("downloaded video size does not match manifest")
            if digest != manifest.video.sha256:
                raise ValueError("downloaded video checksum does not match manifest")
            os.replace(temporary, destination)
            checksum_temporary = checksum_file.with_suffix(".tmp")
            checksum_temporary.write_text(digest + "\n", encoding="ascii")
            os.replace(checksum_temporary, checksum_file)
            return destination
        finally:
            temporary.unlink(missing_ok=True)

    def _create_thumbnail(self, video: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".thumbnail-{uuid4().hex}.jpg")
        command = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            "0.5",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            "scale=640:-2:force_original_aspect_ratio=decrease",
            "-q:v",
            "3",
            str(temporary),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            raise ComponentUnavailable("FFmpeg is unavailable for thumbnails") from exc
        if completed.returncode != 0 or not temporary.is_file():
            message = completed.stderr.strip() or "FFmpeg did not create a thumbnail"
            raise ValueError(f"thumbnail generation failed: {message}")
        os.replace(temporary, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
