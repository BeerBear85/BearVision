"""OpenCV preview-stream frame source."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any
from uuid import uuid4

from bearvision.ports import ComponentUnavailable, VideoFrame


class OpenCvPreviewFrameSource:
    """Read a preview stream without blocking the orchestration event loop."""

    def __init__(
        self,
        clock: Any,
        *,
        max_fps: int = 30,
        queue_size: int = 5,
        drain_old_frames: bool = True,
    ) -> None:
        self.clock = clock
        self.max_fps = max_fps
        self.drain_old_frames = drain_old_frames
        self._queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=queue_size)
        self._capture: Any | None = None
        self._reader: asyncio.Task[None] | None = None
        self._closed = True

    async def open(self, preview_source: str) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - production dependency
            raise ComponentUnavailable("opencv-python-headless is required for preview frames") from exc
        capture = await asyncio.to_thread(cv2.VideoCapture, preview_source)
        self._capture = capture
        if not capture.isOpened():
            await asyncio.to_thread(capture.release)
            self._capture = None
            raise ComponentUnavailable(f"cannot open preview stream: {preview_source}")
        self._closed = False
        self._reader = asyncio.create_task(self._read_frames())

    async def close(self) -> None:
        self._closed = True
        if self._reader is not None:
            self._reader.cancel()
            with suppress(asyncio.CancelledError):
                await self._reader
            self._reader = None
        if self._capture is not None:
            await asyncio.to_thread(self._capture.release)
            self._capture = None

    async def _read_frames(self) -> None:
        minimum_period_s = 1.0 / self.max_fps
        while not self._closed and self._capture is not None:
            started = self.clock.monotonic()
            ok, pixels = await asyncio.to_thread(self._capture.read)
            if not ok:
                raise ComponentUnavailable("preview stream stopped producing frames")
            frame = VideoFrame(
                frame_id=f"frame-{uuid4().hex}",
                observed_at_monotonic_s=self.clock.monotonic(),
                width_px=int(pixels.shape[1]),
                height_px=int(pixels.shape[0]),
                payload=pixels,
            )
            if self._queue.full() and self.drain_old_frames:
                with suppress(asyncio.QueueEmpty):
                    self._queue.get_nowait()
                    self._queue.task_done()
            await self._queue.put(frame)
            remaining = minimum_period_s - (self.clock.monotonic() - started)
            if remaining > 0:
                await self.clock.sleep(remaining)

    async def frames(self):
        while not self._closed:
            frame = await self._queue.get()
            try:
                yield frame
            finally:
                self._queue.task_done()
