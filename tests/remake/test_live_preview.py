import asyncio
from pathlib import Path
import sys
from threading import Event
from types import ModuleType, SimpleNamespace

import pytest

from bearvision.adapters.opencv_frames import (
    JpegPreviewPublisher,
    OpenCvPreviewFrameSource,
)
from bearvision.adapters import SystemClock
from bearvision.ports import VideoFrame
from bearvision.simulation import VirtualClock


def test_jpeg_preview_publisher_writes_atomic_throttled_frames(tmp_path: Path) -> None:
    async def exercise() -> None:
        clock = VirtualClock()
        encoded: list[object] = []

        def encode(payload: object, quality: int) -> bytes:
            encoded.append((payload, quality))
            return b"\xff\xd8preview\xff\xd9"

        destination = tmp_path / "live-preview.jpg"
        publisher = JpegPreviewPublisher(
            clock,
            destination,
            max_fps=4,
            jpeg_quality=70,
            encoder=encode,
        )
        frame = VideoFrame("frame-1", 0, 2, 1, object())

        await publisher.publish(frame)
        await publisher.publish(frame)
        assert destination.read_bytes() == b"\xff\xd8preview\xff\xd9"
        assert len(encoded) == 1

        clock.advance_by(0.25)
        await publisher.publish(frame)
        assert len(encoded) == 2
        assert list(tmp_path.glob("*.tmp")) == []

    asyncio.run(exercise())


def test_preview_source_discards_buffered_frames_and_yields_the_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_drained = Event()
    released = Event()

    class Capture:
        def __init__(self, _source: str) -> None:
            self.frame_number = 0

        def isOpened(self) -> bool:
            return True

        def read(self):
            if self.frame_number < 10:
                self.frame_number += 1
                return True, SimpleNamespace(
                    shape=(1, 1), frame_number=self.frame_number
                )
            source_drained.set()
            released.wait(timeout=1)
            return False, None

        def release(self) -> None:
            released.set()

    cv2 = ModuleType("cv2")
    cv2.VideoCapture = Capture
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    async def exercise() -> None:
        source = OpenCvPreviewFrameSource(SystemClock(), max_fps=1)
        await source.open("udp://camera-preview")
        try:
            assert await asyncio.to_thread(source_drained.wait, 0.25)
            frame = await asyncio.wait_for(anext(source.frames()), timeout=1)
            assert frame.payload.frame_number == 10
        finally:
            await source.close()

    asyncio.run(exercise())
