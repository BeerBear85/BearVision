"""Recorded-video adapters used by hybrid regression scenarios."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
import re
from typing import Any

from bearvision.contracts import CaptureRequest, MediaAsset
from bearvision.ports import (
    CapturedMedia,
    ComponentUnavailable,
    InvalidComponentData,
    VideoClipper,
    VideoFrame,
)

from .adapters import VirtualClock


class RecordedVideoCamera:
    """Treat a checked-in video as both preview and deterministic captured media."""

    def __init__(
        self,
        path: Path,
        clock: VirtualClock,
        *,
        clipper: VideoClipper,
        capture_dir: Path,
    ) -> None:
        self.path = path.resolve()
        self.clock = clock
        self.clipper = clipper
        self.capture_dir = capture_dir.resolve()
        self.connected = False
        self.previewing = False
        self.captures: dict[str, CapturedMedia] = {}

    async def connect(self) -> None:
        if not self.path.is_file():
            raise ComponentUnavailable(f"recorded video does not exist: {self.path}")
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False
        self.previewing = False

    async def start_preview(self) -> str:
        if not self.connected:
            raise ComponentUnavailable("recorded video camera is disconnected")
        self.previewing = True
        return str(self.path)

    async def stop_preview(self) -> None:
        self.previewing = False

    async def capture(self, request: CaptureRequest) -> CapturedMedia:
        if not self.connected:
            raise ComponentUnavailable("recorded video camera is disconnected")
        if request.request_id not in self.captures:
            await self.clock.sleep(request.post_roll_s)
            filename = self._capture_filename(request.request_id)
            start_s = max(0.0, request.requested_at_monotonic_s - request.pre_roll_s)
            duration_s = request.pre_roll_s + request.post_roll_s
            clip = await self.clipper.extract(
                self.path,
                self.capture_dir / filename,
                start_s=start_s,
                duration_s=duration_s,
            )
            self.captures[request.request_id] = CapturedMedia(
                asset=MediaAsset(
                    asset_id=f"asset-{request.request_id}",
                    filename=clip.path.name,
                    content_type="video/mp4",
                    size_bytes=clip.path.stat().st_size,
                    created_at_utc=self.clock.utc_now(),
                ),
                local_path=clip.path,
            )
        return self.captures[request.request_id]

    @staticmethod
    def _capture_filename(request_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", request_id).strip(".-")
        if not safe:
            raise InvalidComponentData("capture request ID cannot form a safe file name")
        if safe != request_id or len(safe) > 100:
            digest = hashlib.sha256(request_id.encode()).hexdigest()[:8]
            safe = f"{safe[:80]}-{digest}"
        return f"{safe}.mp4"


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
            raise ComponentUnavailable("opencv-python-headless is required") from exc
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
            raise ComponentUnavailable("opencv-python-headless is required") from exc
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
