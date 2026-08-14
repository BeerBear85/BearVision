"""Build anonymous, versioned queue packages from an Edge clip."""

from __future__ import annotations

from datetime import datetime
import hashlib

from bearvision.contracts import (
    BearTagJobObservation,
    EdgeJobManifest,
    JobVideo,
    TagObservation,
)
from bearvision.ports import CapturedMedia


def build_edge_job(
    *,
    job_id: str,
    edge_device_id: str,
    created_at: datetime,
    capture_started_at: datetime,
    capture_ended_at: datetime,
    clip_start_monotonic_s: float,
    video: CapturedMedia,
    observations: tuple[TagObservation, ...],
) -> tuple[EdgeJobManifest, tuple[BearTagJobObservation, ...]]:
    if video.content is not None:
        content = video.content
    else:
        assert video.local_path is not None
        content = video.local_path.read_bytes()
    manifest = EdgeJobManifest(
        jobId=job_id,
        edgeDeviceId=edge_device_id,
        createdAt=created_at,
        captureStartedAt=capture_started_at,
        captureEndedAt=capture_ended_at,
        video=JobVideo(
            filename=video.asset.filename,
            mimeType=video.asset.content_type,
            sizeBytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        ),
    )
    job_observations = tuple(
        BearTagJobObservation(
            bearTagId=item.tag_id,
            offsetMs=round((item.observed_at_monotonic_s - clip_start_monotonic_s) * 1000),
            rssiDbm=item.rssi_dbm,
            accelerationMps2=item.acceleration_mps2,
        )
        for item in observations
    )
    return manifest, job_observations
