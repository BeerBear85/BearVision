"""OpenCV preview-stream frame source."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from bearvision.ports import ComponentUnavailable, VideoFrame


class JpegPreviewPublisher:
    """Publish throttled JPEG snapshots for the local Edge Control server."""

    def __init__(
        self,
        clock: Any,
        destination: str | Path,
        *,
        max_fps: float = 4,
        jpeg_quality: int = 70,
        encoder: Callable[[Any, int], bytes] | None = None,
    ) -> None:
        if max_fps <= 0:
            raise ValueError("max_fps must be positive")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        self.clock = clock
        self.destination = Path(destination)
        self.minimum_period_s = 1.0 / max_fps
        self.jpeg_quality = jpeg_quality
        self.encoder = encoder or self._encode_jpeg
        self._last_published_at: float | None = None

    @staticmethod
    def _encode_jpeg(pixels: Any, quality: int) -> bytes:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - production dependency
            raise ComponentUnavailable("opencv-python is required for live preview") from exc
        ok, encoded = cv2.imencode(
            ".jpg", pixels, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            raise ComponentUnavailable("could not encode live preview frame")
        return encoded.tobytes()

    @staticmethod
    def _atomic_write(destination: Path, payload: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_bytes(payload)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    async def publish(self, frame: VideoFrame) -> None:
        now = self.clock.monotonic()
        if (
            self._last_published_at is not None
            and now - self._last_published_at < self.minimum_period_s
        ):
            return
        encoded = await asyncio.to_thread(
            self.encoder, frame.payload, self.jpeg_quality
        )
        await asyncio.to_thread(self._atomic_write, self.destination, encoded)
        self._last_published_at = now


class OpenCvPreviewFrameSource:
    """Read a preview stream without blocking the orchestration event loop."""

    def __init__(
        self,
        clock: Any,
        *,
        max_fps: int = 30,
        queue_size: int = 1,
        drain_old_frames: bool = True,
        preview_frame_path: str | Path | None = None,
        preview_fps: float = 4,
    ) -> None:
        self.clock = clock
        self.max_fps = max_fps
        self.drain_old_frames = drain_old_frames
        self._queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=queue_size)
        self._capture: Any | None = None
        self._reader: asyncio.Task[None] | None = None
        self._closed = True
        self._preview_publisher = (
            JpegPreviewPublisher(clock, preview_frame_path, max_fps=preview_fps)
            if preview_frame_path is not None
            else None
        )

    async def open(self, preview_source: str) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - production dependency
            raise ComponentUnavailable("opencv-python is required for preview frames") from exc
        capture = await asyncio.to_thread(cv2.VideoCapture, preview_source)
        self._capture = capture
        if not capture.isOpened():
            await asyncio.to_thread(capture.release)
            self._capture = None
            raise ComponentUnavailable(f"cannot open preview stream: {preview_source}")
        buffer_property = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
        if buffer_property is not None:
            with suppress(Exception):
                await asyncio.to_thread(capture.set, buffer_property, 1)
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
        while not self._closed and self._capture is not None:
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
            if self._preview_publisher is not None:
                await self._preview_publisher.publish(frame)
            while self.drain_old_frames and not self._queue.empty():
                with suppress(asyncio.QueueEmpty):
                    self._queue.get_nowait()
                    self._queue.task_done()
            await self._queue.put(frame)

    async def frames(self):
        minimum_period_s = 1.0 / self.max_fps
        last_yielded_at: float | None = None
        while not self._closed:
            if last_yielded_at is not None:
                remaining = minimum_period_s - (
                    self.clock.monotonic() - last_yielded_at
                )
                if remaining > 0:
                    await self.clock.sleep(remaining)
            frame = await self._queue.get()
            while self.drain_old_frames and not self._queue.empty():
                self._queue.task_done()
                frame = self._queue.get_nowait()
            last_yielded_at = self.clock.monotonic()
            try:
                yield frame
            finally:
                self._queue.task_done()
