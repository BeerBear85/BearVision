"""Public component boundaries for BearVision 3."""

from .errors import (
    ComponentError,
    ComponentTimeout,
    ComponentUnavailable,
    InvalidComponentData,
    PermanentComponentError,
)
from .models import CapturedMedia, VideoFrame
from .protocols import Camera, Clock, Detector, FrameSource, Storage, TagRegistry, TagScanner

__all__ = [
    "Camera",
    "CapturedMedia",
    "Clock",
    "ComponentError",
    "ComponentTimeout",
    "ComponentUnavailable",
    "Detector",
    "FrameSource",
    "InvalidComponentData",
    "PermanentComponentError",
    "Storage",
    "TagRegistry",
    "TagScanner",
    "VideoFrame",
]
