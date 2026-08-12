"""Production adapters around existing BearVision component implementations."""

from .box import BoxStorageAdapter
from .gopro import GoProCameraAdapter
from .kbeacon import KBeaconTagScannerAdapter
from .system_clock import SystemClock
from .yolo import YoloDetectorAdapter

__all__ = [
    "BoxStorageAdapter",
    "GoProCameraAdapter",
    "KBeaconTagScannerAdapter",
    "SystemClock",
    "YoloDetectorAdapter",
]
