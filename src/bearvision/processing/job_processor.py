"""Port adapter exposing virtual cameraman as an Edge clip processor."""

from __future__ import annotations

from pathlib import Path

from bearvision.ports import CapturedMedia, PreparedClip

from .virtual_cameraman import VirtualCameramanProcessor


class VirtualCameramanJobProcessor:
    def __init__(self, processor: VirtualCameramanProcessor, output_dir: str | Path) -> None:
        self.processor = processor
        self.output_dir = Path(output_dir)

    async def process(self, media: CapturedMedia) -> PreparedClip:
        processed = await self.processor.process(media, self.output_dir)
        return PreparedClip(
            media=processed.media,
            source_start_offset_s=processed.length_adjustment.source_start_s,
            duration_s=processed.length_adjustment.output_duration_s,
        )
