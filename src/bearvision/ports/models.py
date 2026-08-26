"""In-process values exchanged at component boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from bearvision.contracts import CaptureRequest, MediaAsset


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """A frame plus opaque adapter-owned pixel data."""

    frame_id: str
    observed_at_monotonic_s: float
    width_px: int
    height_px: int
    payload: Any

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        if self.observed_at_monotonic_s < 0:
            raise ValueError("observed_at_monotonic_s must not be negative")
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("frame dimensions must be positive")


@dataclass(frozen=True, slots=True)
class CapturedMedia:
    """Captured media backed by either bytes or a local file."""

    asset: MediaAsset
    content: bytes | None = None
    local_path: Path | None = None

    def __post_init__(self) -> None:
        if (self.content is None) == (self.local_path is None):
            raise ValueError("exactly one media source is required")


class CaptureWindowPrecision(StrEnum):
    """How confidently a capture window is aligned to the monotonic timeline."""

    EXACT = "exact"
    ESTIMATED = "estimated"


class CaptureWindowBasis(StrEnum):
    """Evidence used to place a capture window on the monotonic timeline."""

    DETECTION_REQUEST = "detection_request"
    SIMULATED_MEDIA_TIMELINE = "simulated_media_timeline"
    CAMERA_COMMAND_TIMING = "camera_command_timing"
    CAMERA_COMMAND_TIMING_AND_MEDIA_DURATION = "camera_command_timing_and_media_duration"


@dataclass(frozen=True, slots=True)
class CaptureWindow:
    """One requested or delivered interval on the process monotonic timeline."""

    start_monotonic_s: float
    end_monotonic_s: float
    precision: CaptureWindowPrecision
    basis: CaptureWindowBasis

    def __post_init__(self) -> None:
        if self.start_monotonic_s < 0:
            raise ValueError("capture window start must not be negative")
        if self.end_monotonic_s <= self.start_monotonic_s:
            raise ValueError("capture window end must be later than its start")

    @property
    def duration_s(self) -> float:
        return self.end_monotonic_s - self.start_monotonic_s


def requested_capture_window(
    request: CaptureRequest,
    *,
    earliest_available_monotonic_s: float,
) -> CaptureWindow:
    """Build the detection-centred request, clamped to available camera history."""

    if earliest_available_monotonic_s < 0:
        raise ValueError("earliest available camera time must not be negative")
    if earliest_available_monotonic_s > request.requested_at_monotonic_s:
        raise ValueError("camera media starts after the requested detection")
    return CaptureWindow(
        start_monotonic_s=max(
            earliest_available_monotonic_s,
            request.requested_at_monotonic_s - request.pre_roll_s,
        ),
        end_monotonic_s=request.requested_at_monotonic_s + request.post_roll_s,
        precision=CaptureWindowPrecision.EXACT,
        basis=CaptureWindowBasis.DETECTION_REQUEST,
    )


@dataclass(frozen=True, slots=True)
class CapturedClip:
    """Unmodified camera media plus requested and delivered timing evidence."""

    request_id: str
    media: CapturedMedia
    requested_window: CaptureWindow
    actual_window: CaptureWindow

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("capture request id must not be empty")


@dataclass(frozen=True, slots=True)
class ExtractedClip:
    """Validated local clip produced from a longer source asset."""

    path: Path
    start_s: float
    duration_s: float
    width_px: int
    height_px: int
    has_audio: bool

    def __post_init__(self) -> None:
        if self.start_s < 0:
            raise ValueError("clip start must not be negative")
        if self.duration_s <= 0:
            raise ValueError("clip duration must be positive")
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("clip dimensions must be positive")


@dataclass(frozen=True, slots=True)
class ProcessingTraceEvent:
    """Immutable trace metadata emitted by a clip processor."""

    kind: str
    payload: dict[str, Any]
    source_offset_s: float | None = None

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("processing trace event kind must not be empty")
        if self.source_offset_s is not None and self.source_offset_s < 0:
            raise ValueError("processing trace source offset must not be negative")


@dataclass(frozen=True, slots=True)
class PreparedClip:
    """Processed media and its retained interval within the source clip."""

    media: CapturedMedia
    source_start_offset_s: float
    duration_s: float
    trace_events: tuple[ProcessingTraceEvent, ...] = ()

    def __post_init__(self) -> None:
        if self.source_start_offset_s < 0:
            raise ValueError("source_start_offset_s must not be negative")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
