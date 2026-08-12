"""Detector port adapter for the existing DnnHandler."""

from __future__ import annotations

import asyncio
from typing import Any

from bearvision.contracts import BoundingBox, PersonDetection
from bearvision.ports import InvalidComponentData, VideoFrame

from ._errors import translated_error


class YoloDetectorAdapter:
    def __init__(self, handler: Any) -> None:
        self.handler = handler

    async def detect(self, frame: VideoFrame) -> tuple[PersonDetection, ...]:
        try:
            result = await asyncio.to_thread(self.handler.find_person, frame.payload)
            if not isinstance(result, (list, tuple)) or len(result) != 2:
                raise InvalidComponentData("DnnHandler must return [boxes, confidences]")
            boxes, confidences = result
            if len(boxes) != len(confidences):
                raise InvalidComponentData("DnnHandler returned mismatched boxes and confidences")
            detections = []
            for box, confidence in zip(boxes, confidences):
                if len(box) != 4 or float(box[2]) <= 0 or float(box[3]) <= 0:
                    raise InvalidComponentData(f"invalid person bounding box: {box}")
                detections.append(
                    PersonDetection(
                        frame_id=frame.frame_id,
                        observed_at_monotonic_s=frame.observed_at_monotonic_s,
                        bounding_box=BoundingBox(
                            x_px=max(0, float(box[0])),
                            y_px=max(0, float(box[1])),
                            width_px=float(box[2]),
                            height_px=float(box[3]),
                        ),
                        confidence=float(confidence),
                    )
                )
            return tuple(detections)
        except Exception as exc:
            raise translated_error(exc, "run YOLO person detection") from exc
