"""Public component boundaries for BearVision 3."""

from .errors import (
    ComponentError,
    ComponentTimeout,
    ComponentUnavailable,
    InvalidComponentData,
    PermanentComponentError,
)
from .models import CapturedMedia, ExtractedClip, PreparedClip, VideoFrame
from .protocols import (
    Camera,
    ClipProcessor,
    Clock,
    Detector,
    FrameSource,
    JobQueue,
    ManagedJobQueue,
    Storage,
    TagRegistry,
    TagScanner,
    VideoClipper,
)

__all__ = [
    "Camera",
    "CapturedMedia",
    "ClipProcessor",
    "Clock",
    "ComponentError",
    "ComponentTimeout",
    "ComponentUnavailable",
    "Detector",
    "ExtractedClip",
    "FrameSource",
    "InvalidComponentData",
    "JobQueue",
    "ManagedJobQueue",
    "PermanentComponentError",
    "PreparedClip",
    "Storage",
    "TagRegistry",
    "TagScanner",
    "VideoFrame",
    "VideoClipper",
]
