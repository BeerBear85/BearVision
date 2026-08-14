"""Production adapters around existing BearVision component implementations."""

from .box import BoxStorageAdapter
from .box_queue import BoxJobQueue
from .ffmpeg import FfmpegVideoClipper
from .gopro import GoProCameraAdapter
from .kbeacon import BleakKBeaconSource, KBeaconTagScannerAdapter
from .opencv_frames import OpenCvPreviewFrameSource
from .system_clock import SystemClock
from .yolo import YoloDetectorAdapter

__all__ = [
    "BleakKBeaconSource",
    "BoxStorageAdapter",
    "BoxJobQueue",
    "FfmpegVideoClipper",
    "GoProCameraAdapter",
    "KBeaconTagScannerAdapter",
    "OpenCvPreviewFrameSource",
    "SystemClock",
    "YoloDetectorAdapter",
]
