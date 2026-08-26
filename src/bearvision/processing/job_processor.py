"""Port adapter exposing virtual cameraman as an Edge clip processor."""

from __future__ import annotations

from pathlib import Path

from bearvision.ports import CapturedMedia, PreparedClip, ProcessingTraceEvent

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
            trace_events=(
                ProcessingTraceEvent(
                    kind="virtual_cameraman_completed",
                    payload={
                        "source_filename": media.asset.filename,
                        "processed_filename": processed.media.asset.filename,
                        "tracking_filename": processed.metadata_path.name,
                        "debug_video_filename": processed.debug_video_path.name,
                        "source_size_bytes": processed.source_size_bytes,
                        "processed_size_bytes": processed.processed_size_bytes,
                        "size_reduction_ratio": processed.reduction_ratio,
                        "output_width_px": self.processor.config.output_width_px,
                        "output_height_px": self.processor.config.output_height_px,
                        "state_estimator": "kalman_rts_smoother",
                        "camera_path": "zero_phase_butterworth",
                        "length_adjustment": processed.length_adjustment.to_dict(),
                    },
                ),
                *(
                    ProcessingTraceEvent(
                        kind="tracking_observation",
                        source_offset_s=tracking_frame.source_at_s,
                        payload={
                            **tracking_frame.to_dict(),
                            "coordinate_space": {
                                "width_px": processed.source_width_px,
                                "height_px": processed.source_height_px,
                            },
                        },
                    )
                    for tracking_frame in processed.tracking_frames
                ),
            ),
        )
