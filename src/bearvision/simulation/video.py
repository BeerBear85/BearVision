"""Recorded preview frames used by hybrid regression scenarios."""

from __future__ import annotations

import asyncio
from typing import Any

from bearvision.ports import (
    ComponentUnavailable,
    VideoFrame,
)

class RecordedVideoFrameSource:
    """Read stable media timestamps and pixels from a video without wall-clock pacing."""

    def __init__(self, *, sample_fps: float) -> None:
        self.sample_fps = sample_fps
        self._capture: Any | None = None
        self._closed = True

    async def open(self, preview_source: str) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional Edge dependency
            raise ComponentUnavailable("opencv-python is required") from exc
        capture = await asyncio.to_thread(cv2.VideoCapture, preview_source)
        if not capture.isOpened():
            await asyncio.to_thread(capture.release)
            raise ComponentUnavailable(f"cannot open recorded video: {preview_source}")
        self._capture = capture
        self._closed = False

    async def close(self) -> None:
        self._closed = True
        if self._capture is not None:
            await asyncio.to_thread(self._capture.release)
            self._capture = None

    async def frames(self):
        if self._capture is None:
            raise ComponentUnavailable("recorded video source is not open")
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional Edge dependency
            raise ComponentUnavailable("opencv-python is required") from exc
        source_fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0:
            raise ComponentUnavailable("recorded video reports an invalid frame rate")
        sample_step = max(1, round(source_fps / self.sample_fps))
        frame_index = 0
        while not self._closed:
            ok, pixels = await asyncio.to_thread(self._capture.read)
            if not ok:
                break
            if frame_index % sample_step == 0:
                at_s = frame_index / source_fps
                yield VideoFrame(
                    frame_id=f"video-frame-{frame_index}",
                    observed_at_monotonic_s=at_s,
                    width_px=int(pixels.shape[1]),
                    height_px=int(pixels.shape[0]),
                    payload=pixels,
                )
            frame_index += 1
