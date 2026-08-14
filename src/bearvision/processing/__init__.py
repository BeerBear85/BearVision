"""Edge-side media post-processing."""

from bearvision.config import VirtualCameramanConfig

from .job_processor import VirtualCameramanJobProcessor

from .virtual_cameraman import (
    ClipLengthAdjustment,
    KalmanPositionTracker,
    KalmanRtsSmoother,
    PositionMeasurement,
    ProcessedClip,
    SmoothedPosition,
    TrackingFrame,
    VirtualCameramanProcessor,
    ZeroPhaseButterworthCameraSmoother,
    calculate_length_adjustment,
)

__all__ = [
    "ClipLengthAdjustment",
    "KalmanPositionTracker",
    "KalmanRtsSmoother",
    "PositionMeasurement",
    "ProcessedClip",
    "SmoothedPosition",
    "TrackingFrame",
    "VirtualCameramanConfig",
    "VirtualCameramanJobProcessor",
    "VirtualCameramanProcessor",
    "ZeroPhaseButterworthCameraSmoother",
    "calculate_length_adjustment",
]
