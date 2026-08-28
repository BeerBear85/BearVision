"""Pure package and snapshot rules shared by JobQueue adapters."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from bearvision.contracts import (
    BearTagJobObservation,
    EdgeJobManifest,
    JobResultManifest,
)
from bearvision.ports import CapturedMedia


QUEUE_STATES = ("ready", "processing", "processed", "unresolved", "failed")


def media_bytes(media: CapturedMedia) -> bytes:
    if media.content is not None:
        return media.content
    assert media.local_path is not None
    return media.local_path.read_bytes()


def serialize_manifest(manifest: EdgeJobManifest) -> bytes:
    return (manifest.model_dump_json(by_alias=True, indent=2) + "\n").encode()


def serialize_observations(
    observations: tuple[BearTagJobObservation, ...],
) -> bytes:
    return "".join(
        json.dumps(item.model_dump(mode="json", by_alias=True), separators=(",", ":"))
        + "\n"
        for item in observations
    ).encode()


def serialize_result(result: JobResultManifest) -> bytes:
    return (result.model_dump_json(by_alias=True, indent=2) + "\n").encode()


def job_package_files(
    manifest: EdgeJobManifest,
    video: CapturedMedia,
    observations: tuple[BearTagJobObservation, ...],
) -> tuple[tuple[str, bytes], ...]:
    """Return a complete package in publication order, with READY strictly last."""

    return (
        (manifest.video.filename, media_bytes(video)),
        ("manifest.json", serialize_manifest(manifest)),
        (manifest.observations_filename, serialize_observations(observations)),
        ("READY", b""),
    )


def validate_result_destination(
    result: JobResultManifest,
    user_id: UUID | None,
) -> None:
    if result.status == "processed":
        if user_id is None:
            raise ValueError("processed result requires user id")
        if result.selected_user_id != user_id:
            raise ValueError("processed result user id does not match destination")
    elif user_id is not None:
        raise ValueError("non-processed result must not have a destination user id")


def normalize_queue_snapshot(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize provider-discovered jobs to one ordered administrative view."""

    normalized: list[dict[str, Any]] = []
    for source in jobs:
        item = dict(source)
        job_id = item.get("jobId")
        status = item.get("status")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("queue snapshot job requires jobId")
        if status not in QUEUE_STATES:
            raise ValueError(f"invalid queue snapshot status: {status!r}")
        normalized.append(item)
    state_order = {state: index for index, state in enumerate(QUEUE_STATES)}
    normalized.sort(key=lambda item: (state_order[item["status"]], item["jobId"]))
    return {
        "counts": {
            state: sum(item["status"] == state for item in normalized)
            for state in QUEUE_STATES
        },
        "jobs": normalized,
    }
