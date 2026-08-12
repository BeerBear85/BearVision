"""In-process values exchanged at component boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bearvision.contracts import MediaAsset


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
