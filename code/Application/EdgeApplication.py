"""
Edge Application for BearVision system.

This module provides the core EdgeApplication class that coordinates all
edge device functionality including GoPro control, YOLO detection,
BLE beacon monitoring, and status management.
"""

import logging
import threading
import asyncio
import time
from pathlib import Path
import sys
from typing import Optional, List, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1] / "modules"
sys.path.append(str(MODULE_DIR))

from ConfigurationHandler import ConfigurationHandler
from GoProController import GoProController
from DnnHandler import DnnHandler
from ble_beacon_handler import BleBeaconHandler

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


class EdgeApplication:
    """
    Main Edge Application class that coordinates all edge device functionality.

    This class serves as the central coordinator for:
    - GoPro camera connection and preview
    - YOLO person detection
    - BLE beacon monitoring
    - Status management and callbacks
    """

    def __init__(self, status_callback: Optional[Callable[[SystemStatus], None]] = None,
                 detection_callback: Optional[Callable[[DetectionResult], None]] = None,
                 ble_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                 log_callback: Optional[Callable[[str, str], None]] = None):
        """
        Initialize the Edge Application.

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
        """
        self.status_callback = status_callback
        self.detection_callback = detection_callback
        self.ble_callback = ble_callback
        self.log_callback = log_callback

        # Core components
        self.gopro_controller: Optional[GoProController] = None
        self.dnn_handler: Optional[DnnHandler] = None
        self.ble_handler: Optional[BleBeaconHandler] = None

        # Threading and async components
        self.preview_thread: Optional[threading.Thread] = None
        self.ble_thread: Optional[threading.Thread] = None
        self.detection_thread: Optional[threading.Thread] = None
        self.motion_event = asyncio.Event()

        # Status tracking
        self.status = SystemStatus()
        self.running = False
        self.initialized = False

        # Configuration
        self.config_loaded = False
        self.yolo_model = "yolov8n"  # Default YOLO model

    def _log(self, level: str, message: str) -> None:
        """Internal logging with optional callback."""
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

    def _update_status(self, **kwargs) -> None:
        """Update system status and trigger callback."""
        for key, value in kwargs.items():
            if hasattr(self.status, key):
                setattr(self.status, key, value)

        if self.status_callback:
            self.status_callback(self.status)

    def initialize(self, config_path: Optional[str] = None) -> bool:
        """
        Initialize the Edge Application system.

        Parameters
        ----------
        config_path : str, optional
            Path to configuration file. If None, uses default.

        Returns
        -------
        bool
            True if initialization successful, False otherwise
        """
        try:
            self._log("info", "Initializing Edge Application...")
            self._update_status(overall_status=EdgeStatus.INITIALIZING)

            # Load configuration
            if not self._load_config(config_path):
                return False

            # Initialize YOLO detection
            if not self._initialize_yolo():
                return False

            # Initialize BLE handler
            if not self._initialize_ble():
                return False

            self.initialized = True
            self._update_status(overall_status=EdgeStatus.READY)
            self._log("info", "Edge Application initialized successfully")
            return True

        except Exception as e:
            self._log("error", f"Failed to initialize Edge Application: {str(e)}")
            self._update_status(overall_status=EdgeStatus.ERROR)
            return False

    def _load_config(self, config_path: Optional[str] = None) -> bool:
        """Load configuration file."""
        try:
            if config_path is None:
                config_path = Path(__file__).resolve().parents[2] / "config.ini"

            ConfigurationHandler.read_config_file(str(config_path))
            self.config_loaded = True
            self._log("info", f"Configuration loaded from {config_path}")
            return True

        except Exception as e:
            self._log("error", f"Failed to load configuration: {str(e)}")
            return False

    def _initialize_yolo(self) -> bool:
        """Initialize YOLO detection system."""
        try:
            self.dnn_handler = DnnHandler(self.yolo_model)
            self.dnn_handler.init()
            self._update_status(yolo_active=True)
            self._log("info", f"YOLO {self.yolo_model} model initialized")
            return True

        except Exception as e:
            self._log("error", f"Failed to initialize YOLO: {str(e)}")
            return False

    def _initialize_ble(self) -> bool:
        """Initialize BLE beacon handler."""
        try:
            self.ble_handler = BleBeaconHandler()
            self._log("info", "BLE beacon handler initialized")
            return True

        except Exception as e:
            self._log("error", f"Failed to initialize BLE: {str(e)}")
            return False

    def connect_gopro(self) -> bool:
        """
        Connect to GoPro camera.

        Returns
        -------
        bool
            True if connection successful, False otherwise
        """
        try:
            self._log("info", "Connecting to GoPro...")
            self.gopro_controller = GoProController()
            self.gopro_controller.connect()
            self.gopro_controller.configure()

            self._update_status(gopro_connected=True)
            self._log("info", "GoPro connected and configured successfully")
            return True

        except Exception as e:
            self._log("error", f"Failed to connect to GoPro: {str(e)}")
            self._update_status(gopro_connected=False)
            return False

    def start_preview(self) -> bool:
        """
        Start GoPro preview stream.

        Returns
        -------
        bool
            True if preview started successfully, False otherwise
        """
        if not self.gopro_controller:
            self._log("error", "GoPro not connected")
            return False

        try:
            self.gopro_controller.start_preview()
            self._update_status(preview_active=True)
            self._log("info", "GoPro preview started")
            return True

        except Exception as e:
            self._log("error", f"Failed to start preview: {str(e)}")
            return False

    def stop_preview(self) -> bool:
        """
        Stop GoPro preview stream.

        Returns
        -------
        bool
            True if preview stopped successfully, False otherwise
        """
        if not self.gopro_controller:
            return True

        try:
            self.gopro_controller.stop_preview()
            self._update_status(preview_active=False)
            self._log("info", "GoPro preview stopped")
            return True

        except Exception as e:
            self._log("error", f"Failed to stop preview: {str(e)}")
            return False

    def start_ble_logging(self) -> bool:
        """
        Start BLE beacon logging in background thread.

        Returns
        -------
        bool
            True if BLE logging started successfully, False otherwise
        """
        if not self.ble_handler:
            self._log("error", "BLE handler not initialized")
            return False

        try:
            self.ble_thread = threading.Thread(
                target=self._ble_worker,
                name="ble_worker",
                daemon=True
            )
            self.ble_thread.start()

            self._update_status(ble_scanning=True)
            self._log("info", "BLE beacon logging started")
            return True

        except Exception as e:
            self._log("error", f"Failed to start BLE logging: {str(e)}")
            return False

    def _ble_worker(self) -> None:
        """BLE worker thread function."""
        try:
            # Wrap the BLE handler's async function for the thread
            asyncio.run(self._ble_scan_loop())
        except Exception as e:
            self._log("error", f"BLE worker error: {str(e)}")
            self._update_status(ble_scanning=False)

    async def _ble_scan_loop(self) -> None:
        """Async BLE scanning loop."""
        # Modify the BLE handler to use our callback
        original_process = self.ble_handler.process_advertisements

        async def custom_process():
            while self.running:
                try:
                    advertisement = await self.ble_handler.advertisement_queue.get()
                    if self.ble_callback:
                        self.ble_callback(advertisement)
                    else:
                        # Default processing
                        acc = advertisement['acc_sensor']
                        self._log("debug", f"BLE data: {acc.get_value_string()}")
                    self.ble_handler.advertisement_queue.task_done()
                except Exception as e:
                    self._log("error", f"BLE processing error: {str(e)}")

        # Override the process method temporarily
        self.ble_handler.process_advertisements = custom_process

        # Start the scanning
        await self.ble_handler.start_scan_async(timeout=0.0)

    def start_system(self) -> bool:
        """
        Start the complete Edge system.

        This method coordinates the startup of all subsystems:
        1. Connect to GoPro
        2. Start preview
        3. Start BLE logging
        4. Start YOLO detection on preview feed
        5. Update status to "Looking for wakeboarder"

        Returns
        -------
        bool
            True if all systems started successfully, False otherwise
        """
        if not self.initialized:
            self._log("error", "System not initialized")
            return False

        try:
            self._log("info", "Starting Edge system...")
            self.running = True

            # Connect to GoPro
            if not self.connect_gopro():
                return False

            # Start preview
            if not self.start_preview():
                return False

            # Start BLE logging
            if not self.start_ble_logging():
                return False

            # Start detection processing
            if not self._start_detection_processing():
                return False

            # Update status to looking for wakeboarder
            self._update_status(overall_status=EdgeStatus.LOOKING_FOR_WAKEBOARDER)
            self._log("info", "All systems active - Looking for wakeboarder")

            return True

        except Exception as e:
            self._log("error", f"Failed to start system: {str(e)}")
            self._update_status(overall_status=EdgeStatus.ERROR)
            return False

    def _start_detection_processing(self) -> bool:
        """Start YOLO detection processing on preview feed."""
        try:
            self.detection_thread = threading.Thread(
                target=self._detection_worker,
                name="detection_worker",
                daemon=True
            )
            self.detection_thread.start()

            self._log("info", "YOLO detection processing started")
            return True

        except Exception as e:
            self._log("error", f"Failed to start detection processing: {str(e)}")
            return False

    def _detection_worker(self) -> None:
        """Detection worker thread that processes preview frames."""
        while self.running and self.status.preview_active:
            try:
                # Simulate getting frame from GoPro preview
                # In real implementation, this would get actual frames
                time.sleep(0.1)  # Limit processing rate

                # Placeholder for actual frame processing
                # This would integrate with the preview stream
                self._process_mock_detection()

            except Exception as e:
                self._log("error", f"Detection worker error: {str(e)}")
                break

    def _process_mock_detection(self) -> None:
        """Process mock detection for testing."""
        # This is a placeholder - real implementation would process actual frames
        import random

        if random.random() < 0.05:  # 5% chance of detection
            boxes = [[100, 100, 200, 300]]  # Mock bounding box
            confidences = [0.85]

            result = DetectionResult(
                boxes=boxes,
                confidences=confidences,
                timestamp=time.time()
            )

            if self.detection_callback:
                self.detection_callback(result)

            self._update_status(overall_status=EdgeStatus.MOTION_DETECTED)
            self._log("info", "Wakeboarder detected!")

            # Trigger hindsight clip
            self.trigger_hindsight()

    def trigger_hindsight(self) -> bool:
        """
        Trigger a hindsight clip recording.

        Returns
        -------
        bool
            True if hindsight triggered successfully, False otherwise
        """
        if not self.gopro_controller:
            self._log("error", "GoPro not connected")
            return False

        try:
            self.gopro_controller.start_hindsight_clip()
            self._update_status(hindsight_mode=True, overall_status=EdgeStatus.RECORDING)
            self._log("info", "Hindsight clip triggered")

            # Reset to looking for wakeboarder after a delay
            threading.Timer(5.0, self._reset_to_looking).start()

            return True

        except Exception as e:
            self._log("error", f"Failed to trigger hindsight: {str(e)}")
            return False

    def _reset_to_looking(self) -> None:
        """Reset status back to looking for wakeboarder."""
        self._update_status(
            hindsight_mode=False,
            overall_status=EdgeStatus.LOOKING_FOR_WAKEBOARDER
        )
        self._log("info", "Ready - Looking for wakeboarder")

    def stop_system(self) -> None:
        """Stop the Edge system and cleanup resources."""
        self._log("info", "Stopping Edge system...")
        self.running = False

        # Stop preview
        self.stop_preview()

        # Disconnect GoPro
        if self.gopro_controller:
            try:
                self.gopro_controller.disconnect()
                self._log("info", "GoPro disconnected")
            except Exception as e:
                self._log("warning", f"Error disconnecting GoPro: {str(e)}")

        # Update status
        self._update_status(
            overall_status=EdgeStatus.STOPPED,
            gopro_connected=False,
            preview_active=False,
            ble_scanning=False,
            hindsight_mode=False,
            recording=False
        )

        self._log("info", "Edge system stopped")

    def get_status(self) -> SystemStatus:
        """
        Get current system status.

        Returns
        -------
        SystemStatus
            Current system status
        """
        return self.status

    def is_running(self) -> bool:
        """
        Check if the system is running.

        Returns
        -------
        bool
            True if system is running, False otherwise
        """
        return self.running and self.initialized