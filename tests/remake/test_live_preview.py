import asyncio
from pathlib import Path

from bearvision.adapters.opencv_frames import JpegPreviewPublisher
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
