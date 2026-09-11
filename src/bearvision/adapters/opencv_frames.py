"""OpenCV preview-stream frame source."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
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
        self._executor: ThreadPoolExecutor | None = None
        self._release_future: Future[None] | None = None
        self._closed = True
        self._preview_publisher = (
            JpegPreviewPublisher(clock, preview_frame_path, max_fps=preview_fps)
            if preview_frame_path is not None
            else None
        )

    async def open(self, preview_source: str) -> None:
        await self.close()
        self._queue = asyncio.Queue(maxsize=self._queue.maxsize)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preview")
        self._release_future = None
        opening = asyncio.wrap_future(self._executor.submit(self._open_capture, preview_source))
        # A cancelled caller must not lose a capture returned later by native open().
        opening.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        try:
            await asyncio.shield(opening)
        except BaseException:
            self._schedule_release()
            raise
        self._closed = False
        self._reader = asyncio.create_task(self._read_frames())

    def _open_capture(self, preview_source: str) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - production dependency
            raise ComponentUnavailable("opencv-python is required for preview frames") from exc
        # FFmpeg timeouts are open-time parameters, not properties set afterwards.
        # Native reads must return within the readiness cleanup budget (3 seconds).
        capture = cv2.VideoCapture(preview_source, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 20_000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1_000,
        ])
        self._capture = capture
        if not capture.isOpened():
            raise ComponentUnavailable(f"cannot open preview stream: {preview_source}")
        buffer_property = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
        if buffer_property is not None:
            with suppress(Exception):
                capture.set(buffer_property, 1)

    def _release_capture(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    def _schedule_release(self) -> Future[None] | None:
        if self._executor is not None and self._release_future is None:
            executor = self._executor
            # Cancellation stops awaiting native I/O, not the I/O itself.
            # Release on the same worker, behind any still-running open/read.
            self._release_future = executor.submit(self._release_capture)
            self._release_future.add_done_callback(lambda _: executor.shutdown(wait=False))
        return self._release_future

    async def close(self) -> None:
        self._closed = True
        reader, self._reader = self._reader, None
        try:
            if reader is not None:
                reader.cancel()
                # frames() owns reader errors; they must not skip resource cleanup.
                await asyncio.gather(reader, return_exceptions=True)
        finally:
            release = self._schedule_release()
        if release is not None:
            # Even a cancelled close leaves the queued native release intact.
            await asyncio.shield(asyncio.wrap_future(release))
        self._executor = None
        self._release_future = None

    async def _read_frames(self) -> None:
        while not self._closed and self._capture is not None:
            assert self._executor is not None
            ok, pixels = await asyncio.wrap_future(self._executor.submit(self._capture.read))
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
            reader = self._reader
            if reader is None:
                return
            pending_frame = asyncio.create_task(self._queue.get())
            try:
                done, _ = await asyncio.wait(
                    {pending_frame, reader}, return_when=asyncio.FIRST_COMPLETED,
                )
                if self._closed:
                    return
                if pending_frame not in done:
                    await reader  # Surface read failures instead of waiting forever.
                    return
                frame = pending_frame.result()
            finally:
                if not pending_frame.done():
                    pending_frame.cancel()
                    await asyncio.gather(pending_frame, return_exceptions=True)
            while self.drain_old_frames and not self._queue.empty():
                self._queue.task_done()
                frame = self._queue.get_nowait()
            last_yielded_at = self.clock.monotonic()
            try:
                yield frame
            finally:
                self._queue.task_done()
