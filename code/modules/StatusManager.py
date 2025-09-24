"""
Status Management Module

This module provides centralized status tracking, logging, and callback management
for the Edge Application system. It handles system status updates, detection results,
and coordinates callbacks to GUI and other system components.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, Dict, Any, List


logger = logging.getLogger(__name__)


class EdgeStatus(Enum):
    """Edge application status states."""
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    LOOKING_FOR_WAKEBOARDER = "looking_for_wakeboarder"
    MOTION_DETECTED = "motion_detected"
    RECORDING = "recording"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class SystemStatus:
    """System status indicators."""
    overall_status: EdgeStatus = EdgeStatus.INITIALIZING
    gopro_connected: bool = False
    preview_active: bool = False
    ble_scanning: bool = False
    yolo_active: bool = False
    hindsight_mode: bool = False
    recording: bool = False


@dataclass
class DetectionResult:
    """YOLO detection result."""
    boxes: List[List[int]]
    confidences: List[float]
    timestamp: float


class StatusManager:
    """
    Centralized status and callback management for Edge Application.

    This class handles:
    - System status tracking and updates
    - Callback coordination to GUI and other components
    - Logging with optional callback integration
    - Status change notifications
    """

    def __init__(self,
                 status_callback: Optional[Callable[[SystemStatus], None]] = None,
                 detection_callback: Optional[Callable[[DetectionResult], None]] = None,
                 ble_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                 log_callback: Optional[Callable[[str, str], None]] = None,
                 frame_callback: Optional[Callable] = None):
        """
        Initialize the Status Manager.

        Parameters
        ----------
        status_callback : Callable[[SystemStatus], None], optional
            Callback function for status updates
        detection_callback : Callable[[DetectionResult], None], optional
            Callback function for YOLO detection results
        ble_callback : Callable[[Dict[str, Any]], None], optional
            Callback function for BLE beacon data
        log_callback : Callable[[str, str], None], optional
            Callback function for log messages (level, message)
        frame_callback : Callable, optional
            Callback function for preview frames
        """
        # Callback functions
        self.status_callback = status_callback
        self.detection_callback = detection_callback
        self.ble_callback = ble_callback
        self.log_callback = log_callback
        self.frame_callback = frame_callback

        # Current system status
        self.status = SystemStatus()

        # Status history for debugging
        self.status_history: List[tuple] = []
        self.max_history_length = 100

    def log(self, level: str, message: str) -> None:
        """
        Log message with optional callback.

        Parameters
        ----------
        level : str
            Log level ('info', 'warning', 'error', 'debug')
        message : str
            Log message
        """
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "debug":
            logger.debug(message)

        if self.log_callback:
            self.log_callback(level, message)

    def update_status(self, **kwargs) -> None:
        """
        Update system status and trigger callback.

        Parameters
        ----------
        **kwargs : dict
            Status attributes to update
        """
        old_status = SystemStatus(**self.status.__dict__)

        # Update status attributes
        for key, value in kwargs.items():
            if hasattr(self.status, key):
                setattr(self.status, key, value)
            else:
                self.log("warning", f"Unknown status attribute: {key}")

        # Track status changes in history
        timestamp = time.time()
        changes = {k: v for k, v in kwargs.items() if hasattr(self.status, k)}
        if changes:
            self.status_history.append((timestamp, changes))

            # Limit history size
            if len(self.status_history) > self.max_history_length:
                self.status_history = self.status_history[-self.max_history_length:]

        # Trigger callback if status changed
        if self.status_callback and self._status_changed(old_status, self.status):
            self.status_callback(self.status)

    def _status_changed(self, old_status: SystemStatus, new_status: SystemStatus) -> bool:
        """Check if status actually changed."""
        return old_status.__dict__ != new_status.__dict__

    def get_status(self) -> SystemStatus:
        """Get current system status."""
        return SystemStatus(**self.status.__dict__)  # Return a copy

    def get_status_history(self) -> List[tuple]:
        """Get status change history."""
        return self.status_history.copy()

    def trigger_detection_callback(self, detection: DetectionResult) -> None:
        """Trigger detection result callback if available."""
        if self.detection_callback:
            self.detection_callback(detection)

    def trigger_ble_callback(self, ble_data: Dict[str, Any]) -> None:
        """Trigger BLE data callback if available."""
        if self.ble_callback:
            self.ble_callback(ble_data)

    def trigger_frame_callback(self, frame) -> None:
        """Trigger frame callback if available."""
        if self.frame_callback:
            try:
                self.frame_callback(frame)
            except Exception as e:
                self.log("error", f"Frame callback error: {e}")

    def is_system_ready(self) -> bool:
        """Check if system is in ready state for operations."""
        return (self.status.overall_status in [EdgeStatus.READY, EdgeStatus.ACTIVE,
                                             EdgeStatus.LOOKING_FOR_WAKEBOARDER] and
                self.status.gopro_connected and
                self.status.yolo_active)

    def is_system_active(self) -> bool:
        """Check if system is actively running operations."""
        return (self.status.overall_status in [EdgeStatus.ACTIVE,
                                             EdgeStatus.LOOKING_FOR_WAKEBOARDER,
                                             EdgeStatus.MOTION_DETECTED,
                                             EdgeStatus.RECORDING])

    def reset_to_ready(self) -> None:
        """Reset system status to ready state."""
        self.update_status(
            overall_status=EdgeStatus.READY,
            recording=False,
            hindsight_mode=False
        )

    def set_error_state(self, error_message: str) -> None:
        """Set system to error state with message."""
        self.update_status(overall_status=EdgeStatus.ERROR)
        self.log("error", error_message)

    def shutdown(self) -> None:
        """Shutdown status manager and clean up."""
        self.update_status(
            overall_status=EdgeStatus.STOPPED,
            gopro_connected=False,
            preview_active=False,
            ble_scanning=False,
            hindsight_mode=False,
            recording=False
        )
        self.log("info", "Status Manager shutdown complete")