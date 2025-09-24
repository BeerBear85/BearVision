"""
Edge Application with State Machine Architecture

This module provides a modular and maintainable architecture for the Edge Application,
centered around a main state machine with supporting threads for asynchronous operations.

The application supports three main states:
- INITIALIZING: Setup all subsystems
- SEARCHING_FOR_WAKEBOARDER: Monitor for person detection
- RECORDING: Actively recording/capturing footage

Background threads handle:
- BLE tag logging (continuous)
- Video post-processing (queued)
- File uploads to cloud storage (queued)
"""

import logging
import threading
import asyncio
import time
import queue
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import sys
import numpy as np
import cv2

# Add module paths
MODULE_DIR = Path(__file__).resolve().parents[1] / "modules"
sys.path.append(str(MODULE_DIR))

from ConfigurationHandler import ConfigurationHandler
from GoProController import GoProController
from DnnHandler import DnnHandler
from ble_beacon_handler import BleBeaconHandler
from GoogleDriveHandler import GoogleDriveHandler
from BoxHandler import BoxHandler
from FullClipExtractor import FullClipExtractor
from TrackerClipExtractor import TrackerClipExtractor

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
                 log_callback: Optional[Callable[[str, str], None]] = None,
                 frame_callback: Optional[Callable[[np.ndarray], None]] = None):
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
        frame_callback : Callable[[np.ndarray], None], optional
            Callback function for preview frames
        """
        self.status_callback = status_callback
        self.detection_callback = detection_callback
        self.ble_callback = ble_callback
        self.log_callback = log_callback
        self.frame_callback = frame_callback

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

        # Preview stream URL
        self.preview_stream_url = None

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

    def _wait_for_gopro_ready(self, max_retries: int = 15, initial_delay: float = 0.5) -> bool:
        """
        Wait for GoPro connection to be fully established.

        Parameters
        ----------
        max_retries : int
            Maximum number of retry attempts
        initial_delay : float
            Initial delay in seconds, doubles with each retry

        Returns
        -------
        bool
            True if GoPro is ready, False if timeout
        """
        import time

        delay = initial_delay
        for attempt in range(max_retries):
            try:
                # Multiple checks to verify GoPro is ready
                gopro = self.gopro_controller._gopro

                # Check 1: Basic connection attributes
                if not hasattr(gopro, '_serial') or not gopro._serial:
                    self._log("debug", f"GoPro serial not available (attempt {attempt + 1})")
                    time.sleep(delay)
                    delay = min(delay * 1.5, 3.0)
                    continue

                # Check 2: Try to get camera status to verify connection is working
                try:
                    status = self.gopro_controller._run_in_thread(gopro.http_command.get_camera_state())
                    if status and hasattr(status, 'data') and status.data:
                        self._log("info", f"GoPro connection ready (attempt {attempt + 1})")
                        return True
                    else:
                        self._log("debug", f"GoPro status check returned empty data (attempt {attempt + 1})")
                except Exception as status_error:
                    self._log("debug", f"GoPro status check failed: {status_error} (attempt {attempt + 1})")

                # Check 3: HTTP settings availability (optional, not required for basic functionality)
                if hasattr(gopro, 'http_settings'):
                    self._log("debug", f"GoPro http_settings available (attempt {attempt + 1})")
                else:
                    self._log("debug", f"GoPro http_settings not yet available (attempt {attempt + 1})")

                time.sleep(delay)
                delay = min(delay * 1.5, 3.0)  # Slower exponential backoff

            except Exception as e:
                self._log("debug", f"Error checking GoPro readiness: {e}")
                time.sleep(delay)
                delay = min(delay * 1.5, 3.0)

        self._log("error", f"GoPro connection not ready after {max_retries} attempts")
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

            # Wait for connection to be fully established with retry logic
            self._log("info", "Verifying GoPro connection...")
            if not self._wait_for_gopro_ready():
                self._log("error", "GoPro connection not ready after waiting")
                return False

            self._log("info", "Configuring GoPro settings...")
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
            # Start preview and get the stream URL
            self.preview_stream_url = self.gopro_controller.start_preview()
            self._update_status(preview_active=True)
            self._log("info", f"GoPro preview started, URL: {self.preview_stream_url}")
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
        3. Enable hindsight mode
        4. Start BLE logging
        5. Start YOLO detection on preview feed
        6. Update status to "Looking for wakeboarder"

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

            # Enable hindsight mode
            if not self.trigger_hindsight():
                self._log("warning", "Failed to enable hindsight mode, continuing without it")

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
            # Ensure the system is marked as running for the detection worker
            if not self.running:
                self.running = True
                self._log("debug", "Setting running=True for detection processing")

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
        import time
        import threading

        cap = None
        stream_url = None

        # Get the preview stream URL from GoPro controller
        try:
            if self.gopro_controller and self.status.preview_active:
                urls_to_try = []

                # First, try the actual URL from GoProController if available
                if self.preview_stream_url:
                    urls_to_try.append(self.preview_stream_url)
                    self._log("debug", f"Using GoPro controller URL: {self.preview_stream_url}")

                # Fallback to common GoPro IP addresses
                fallback_urls = [
                    "udp://172.24.106.51:8554",  # Common wired IP
                    "udp://172.20.110.51:8554",  # Alternative wired IP
                    "udp://172.25.90.51:8554",   # Another alternative
                    "udp://10.5.5.9:8554"       # WiFi IP
                ]

                # Add fallbacks (but avoid duplicates)
                for url in fallback_urls:
                    if url not in urls_to_try:
                        urls_to_try.append(url)

                # Try each URL to see which one works
                for url in urls_to_try:
                    self._log("debug", f"Trying preview stream URL: {url}")
                    try:
                        test_cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

                        if test_cap.isOpened():
                            # Configure capture properties for better streaming
                            test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            test_cap.set(cv2.CAP_PROP_FPS, 30)
                            # Don't force resolution - let GoPro stream at native resolution

                            # Test if we can actually read a frame (with shorter timeout)
                            self._log("debug", f"Testing frame read from {url}...")

                            # Try multiple frame reads to ensure stream is working
                            successful_reads = 0
                            for i in range(3):
                                ret, test_frame = test_cap.read()
                                if ret and test_frame is not None:
                                    successful_reads += 1
                                    self._log("debug", f"Test read {i+1}: SUCCESS, frame shape: {test_frame.shape}")
                                else:
                                    self._log("debug", f"Test read {i+1}: FAILED, ret={ret}")
                                time.sleep(0.1)  # Brief pause between reads

                            if successful_reads >= 1:
                                self._log("info", f"Successfully connected to preview stream: {url} ({successful_reads}/3 test reads)")
                                stream_url = url
                                cap = test_cap

                                # Re-configure for optimal performance
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                cap.set(cv2.CAP_PROP_FPS, 30)
                                break
                            else:
                                self._log("debug", f"Could not read frames from {url}")
                                test_cap.release()
                        else:
                            self._log("debug", f"Could not open {url}")
                            test_cap.release()

                    except Exception as url_error:
                        self._log("debug", f"Error trying {url}: {url_error}")
                        if 'test_cap' in locals():
                            test_cap.release()

                if not cap:
                    self._log("warning", "Could not connect to any GoPro preview stream")

        except Exception as e:
            self._log("error", f"Failed to initialize preview capture: {str(e)}")
            cap = None

        frame_count = 0
        no_frame_count = 0
        last_debug_time = time.time()

        # Main frame processing loop
        self._log("info", f"Starting frame processing loop with stream: {stream_url}")
        self._log("debug", f"Loop conditions: running={self.running}, preview_active={self.status.preview_active}")

        while self.running and self.status.preview_active:
            # Debug loop conditions periodically
            if frame_count % 100 == 0 and frame_count > 0:
                self._log("debug", f"Loop still active: running={self.running}, preview_active={self.status.preview_active}, frames={frame_count}")
            try:
                if cap and cap.isOpened():
                    # Try to read frame
                    ret, frame = cap.read()

                    # Debug: Log first few read attempts
                    if frame_count < 10:
                        self._log("debug", f"Frame read attempt {frame_count + 1}: ret={ret}, frame={'None' if frame is None else frame.shape if frame is not None else 'Invalid'}")

                    if ret and frame is not None:
                        # Successfully read frame
                        frame_count += 1
                        no_frame_count = 0

                        # Create a copy for GUI to avoid threading issues
                        frame_copy = frame.copy()

                        # Run YOLO detection on every 5th frame
                        detection_boxes = []
                        if frame_count % 5 == 0:
                            detection_boxes = self._process_frame_detection_with_boxes(frame_copy)

                        # Draw detection boxes on the frame copy
                        if detection_boxes:
                            frame_copy = self._draw_detection_boxes(frame_copy, detection_boxes)

                        # Send frame to GUI via callback
                        if hasattr(self, 'frame_callback') and self.frame_callback:
                            try:
                                self.frame_callback(frame_copy)
                                if frame_count <= 5 or frame_count % 30 == 0:
                                    self._log("debug", f"Sent frame {frame_count} to GUI callback")
                            except Exception as callback_error:
                                self._log("error", f"Frame callback error: {callback_error}")
                        else:
                            if frame_count <= 5:
                                self._log("warning", "Frame callback not available")

                        # Log success periodically
                        if frame_count == 1 or frame_count % 30 == 0:
                            self._log("info", f"Processing frames successfully, count: {frame_count}")

                    else:
                        no_frame_count += 1

                        # Log frame reading issues
                        if no_frame_count <= 5 or no_frame_count % 50 == 0:
                            self._log("warning", f"Frame read failed: ret={ret} (attempt {no_frame_count})")

                        # Log frame reading issues periodically
                        current_time = time.time()
                        if current_time - last_debug_time > 5.0:  # Every 5 seconds
                            self._log("warning", f"Still no frames from stream after {no_frame_count} attempts")
                            last_debug_time = current_time

                        # Brief pause before retrying
                        time.sleep(0.033)  # ~30fps attempt rate

                else:
                    # If no capture available
                    self._log("error", f"Video capture not available: cap={cap}, isOpened={cap.isOpened() if cap else 'N/A'}")
                    time.sleep(1.0)  # Longer pause when no capture
                    break

            except Exception as e:
                self._log("error", f"Detection worker error: {str(e)}")
                time.sleep(0.1)

        # Log why the loop ended
        self._log("info", f"Frame processing loop ended. Total frames processed: {frame_count}")
        self._log("debug", f"Loop exit conditions: running={self.running}, preview_active={self.status.preview_active}")

        # Cleanup
        if cap:
            cap.release()

    def _process_frame_detection_with_boxes(self, frame: np.ndarray) -> list:
        """Process frame for YOLO detection and return bounding boxes."""
        boxes = []
        try:
            if self.dnn_handler:
                # Run YOLO detection on the frame
                detection_result = self.dnn_handler.find_person(frame)

                if detection_result and len(detection_result) == 2:
                    detected_boxes, confidences = detection_result

                    if detected_boxes and confidences:
                        for i, (box, confidence) in enumerate(zip(detected_boxes, confidences)):
                            if confidence > 0.5:  # Filter low confidence detections
                                # Convert from [x, y, w, h] to [x1, y1, x2, y2]
                                x, y, w, h = box
                                x1, y1, x2, y2 = x, y, x + w, y + h

                                boxes.append({
                                    'coords': [x1, y1, x2, y2],
                                    'confidence': confidence,
                                    'label': 'Person'
                                })

            if boxes:
                # Trigger hindsight clip for motion detection
                self._update_status(overall_status=EdgeStatus.MOTION_DETECTED)
                self._log("info", f"Person detected with {len(boxes)} bounding boxes!")
                self.trigger_hindsight_clip()

        except Exception as e:
            self._log("error", f"Frame detection error: {str(e)}")

        return boxes

    def _draw_detection_boxes(self, frame: np.ndarray, boxes: list) -> np.ndarray:
        """Draw YOLO detection boxes on frame."""
        try:
            import cv2

            for box in boxes:
                coords = box['coords']
                confidence = box['confidence']
                label = box['label']

                # Extract coordinates
                x1, y1, x2, y2 = map(int, coords)

                # Draw green rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label with confidence
                label_text = f"{label}: {confidence:.2f}"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                # Draw label background
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                             (x1 + label_size[0], y1), (0, 255, 0), -1)

                # Draw label text
                cv2.putText(frame, label_text, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        except Exception as e:
            self._log("error", f"Error drawing detection boxes: {str(e)}")

        return frame

    def _process_frame_detection(self, frame: np.ndarray) -> None:
        """Process actual frame for YOLO detection."""
        try:
            if self.dnn_handler:
                # Run YOLO detection on the frame
                detection_result = self.dnn_handler.find_person(frame)
                if detection_result and len(detection_result) == 2:
                    # DnnHandler returns [boxes, confidences] format
                    detected_boxes, confidences = detection_result
                    boxes = detected_boxes  # Already in the right format

                    if boxes:
                        result = DetectionResult(
                            boxes=boxes,
                            confidences=confidences,
                            timestamp=time.time()
                        )

                        if self.detection_callback:
                            self.detection_callback(result)

                        self._update_status(overall_status=EdgeStatus.MOTION_DETECTED)
                        self._log("info", f"Person detected with {len(boxes)} bounding boxes!")

                        # Trigger hindsight clip recording
                        self.trigger_hindsight_clip()

        except Exception as e:
            self._log("error", f"Frame detection error: {str(e)}")

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

            # Trigger hindsight clip recording
            self.trigger_hindsight_clip()

    def trigger_hindsight(self) -> bool:
        """
        Enable hindsight mode on the camera.

        Returns
        -------
        bool
            True if hindsight mode enabled successfully, False otherwise
        """
        if not self.gopro_controller:
            self._log("error", "GoPro not connected")
            return False

        try:
            success = self.gopro_controller.startHindsightMode()
            if success:
                self._update_status(hindsight_mode=True)
                self._log("info", "Hindsight mode enabled")
                return True
            else:
                self._log("warning", "Failed to enable hindsight mode - GoPro settings not available")
                return False

        except Exception as e:
            self._log("error", f"Failed to enable hindsight mode: {str(e)}")
            return False

    def trigger_hindsight_clip(self) -> bool:
        """
        Trigger a hindsight clip recording (for automatic detection).

        Returns
        -------
        bool
            True if hindsight clip triggered successfully, False otherwise
        """
        if not self.gopro_controller:
            self._log("error", "GoPro not connected")
            return False

        try:
            self.gopro_controller.start_hindsight_clip()
            self._update_status(overall_status=EdgeStatus.RECORDING)
            self._log("info", "Hindsight clip triggered")

            # Reset to looking for wakeboarder after a delay
            threading.Timer(5.0, self._reset_to_looking).start()

            return True

        except Exception as e:
            self._log("error", f"Failed to trigger hindsight clip: {str(e)}")
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


class ApplicationState(Enum):
    """Main application states for the state machine."""
    INITIALIZING = "initializing"
    SEARCHING_FOR_WAKEBOARDER = "searching_for_wakeboarder"
    RECORDING = "recording"
    SHUTDOWN = "shutdown"


@dataclass
class VideoFile:
    """Represents a video file to be processed."""
    file_path: Path
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class ProcessedVideo:
    """Represents a processed video ready for upload."""
    original_path: Path
    processed_path: Path
    metadata: Dict[str, Any]
    processing_time: float


class BLELoggerThread(threading.Thread):
    """Continuous BLE tag logging thread."""

    def __init__(self, status_callback: Optional[Callable[[str, Any], None]] = None):
        super().__init__(name="BLELogger", daemon=True)
        self.ble_handler = BleBeaconHandler()
        self.status_callback = status_callback
        self.running = False
        self.ble_data_queue = queue.Queue()

    def start_logging(self):
        """Start BLE logging."""
        self.running = True
        self.start()
        logger.info("BLE logging thread started")

    def stop_logging(self):
        """Stop BLE logging."""
        self.running = False
        logger.info("BLE logging thread stopped")

    def run(self):
        """Main BLE logging loop."""
        logger.info("BLE logging thread running")

        while self.running:
            try:
                # Start BLE scanning for a short duration
                self.ble_handler.start_scan(timeout=1.0)

                # Process any queued BLE data
                while not self.ble_data_queue.empty():
                    try:
                        ble_data = self.ble_data_queue.get_nowait()
                        if self.status_callback:
                            self.status_callback("ble_data", ble_data)
                        self.ble_data_queue.task_done()
                    except queue.Empty:
                        break

                time.sleep(0.1)  # Brief pause between scans

            except Exception as e:
                logger.error(f"BLE logging error: {e}")
                time.sleep(1.0)  # Longer pause on error


class PostProcessorThread(threading.Thread):
    """Video post-processing thread."""

    def __init__(self, video_queue: queue.Queue, processed_queue: queue.Queue,
                 status_callback: Optional[Callable[[str, Any], None]] = None):
        super().__init__(name="PostProcessor", daemon=True)
        self.video_queue = video_queue
        self.processed_queue = processed_queue
        self.status_callback = status_callback
        self.running = False

        # Initialize video processors
        self.full_clip_extractor = FullClipExtractor()
        self.tracker_clip_extractor = TrackerClipExtractor()

    def start_processing(self):
        """Start video processing."""
        self.running = True
        self.start()
        logger.info("Video post-processing thread started")

    def stop_processing(self):
        """Stop video processing."""
        self.running = False
        logger.info("Video post-processing thread stopped")

    def run(self):
        """Main video processing loop."""
        logger.info("Video post-processing thread running")

        while self.running:
            try:
                # Wait for video to process
                video_file = self.video_queue.get(timeout=1.0)

                if video_file is None:  # Shutdown signal
                    break

                logger.info(f"Processing video: {video_file.file_path}")

                if self.status_callback:
                    self.status_callback("processing_started", video_file)

                # Process the video file
                processed_video = self._process_video(video_file)

                if processed_video:
                    self.processed_queue.put(processed_video)
                    logger.info(f"Video processed successfully: {processed_video.processed_path}")

                    if self.status_callback:
                        self.status_callback("processing_completed", processed_video)
                else:
                    logger.error(f"Failed to process video: {video_file.file_path}")

                self.video_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Post-processing error: {e}")
                if not self.video_queue.empty():
                    self.video_queue.task_done()

    def _process_video(self, video_file: VideoFile) -> Optional[ProcessedVideo]:
        """Process a single video file."""
        try:
            start_time = time.time()

            # Determine processing type based on metadata
            use_tracker = video_file.metadata.get("use_tracker", False)

            if use_tracker:
                # Use tracker-based clip extraction
                processed_path = self._process_with_tracker(video_file)
            else:
                # Use full clip extraction
                processed_path = self._process_full_clip(video_file)

            if processed_path and processed_path.exists():
                processing_time = time.time() - start_time

                return ProcessedVideo(
                    original_path=video_file.file_path,
                    processed_path=processed_path,
                    metadata=video_file.metadata.copy(),
                    processing_time=processing_time
                )
            else:
                return None

        except Exception as e:
            logger.error(f"Error processing video {video_file.file_path}: {e}")
            return None

    def _process_with_tracker(self, video_file: VideoFile) -> Optional[Path]:
        """Process video using tracker clip extractor."""
        # This is a placeholder - actual implementation would depend on
        # the specific requirements and API of TrackerClipExtractor
        logger.info(f"Processing with tracker: {video_file.file_path}")

        # For now, return a mock processed path
        output_dir = video_file.file_path.parent / "processed"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"tracker_{video_file.file_path.name}"

        # TODO: Implement actual tracker processing
        # self.tracker_clip_extractor.extract_clips_from_list([clip_spec])

        return output_path

    def _process_full_clip(self, video_file: VideoFile) -> Optional[Path]:
        """Process video using full clip extractor."""
        logger.info(f"Processing full clip: {video_file.file_path}")

        # For now, return a mock processed path
        output_dir = video_file.file_path.parent / "processed"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"full_{video_file.file_path.name}"

        # TODO: Implement actual full clip processing
        # self.full_clip_extractor.extract_clips_from_list([clip_spec])

        return output_path


class UploaderThread(threading.Thread):
    """File upload thread for cloud storage."""

    def __init__(self, processed_queue: queue.Queue,
                 status_callback: Optional[Callable[[str, Any], None]] = None):
        super().__init__(name="Uploader", daemon=True)
        self.processed_queue = processed_queue
        self.status_callback = status_callback
        self.running = False

        # Initialize upload handlers
        try:
            self.google_drive_handler = GoogleDriveHandler()
            self.google_drive_available = True
        except Exception as e:
            logger.warning(f"Google Drive handler not available: {e}")
            self.google_drive_handler = None
            self.google_drive_available = False

        try:
            self.box_handler = BoxHandler()
            self.box_available = True
        except Exception as e:
            logger.warning(f"Box handler not available: {e}")
            self.box_handler = None
            self.box_available = False

    def start_uploading(self):
        """Start file uploading."""
        self.running = True
        self.start()
        logger.info("File upload thread started")

    def stop_uploading(self):
        """Stop file uploading."""
        self.running = False
        logger.info("File upload thread stopped")

    def run(self):
        """Main upload loop."""
        logger.info("File upload thread running")

        while self.running:
            try:
                # Wait for processed video to upload
                processed_video = self.processed_queue.get(timeout=1.0)

                if processed_video is None:  # Shutdown signal
                    break

                logger.info(f"Uploading video: {processed_video.processed_path}")

                if self.status_callback:
                    self.status_callback("upload_started", processed_video)

                # Upload the processed video
                success = self._upload_video(processed_video)

                if success:
                    logger.info(f"Video uploaded successfully: {processed_video.processed_path}")

                    if self.status_callback:
                        self.status_callback("upload_completed", processed_video)
                else:
                    logger.error(f"Failed to upload video: {processed_video.processed_path}")

                self.processed_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Upload error: {e}")
                if not self.processed_queue.empty():
                    self.processed_queue.task_done()

    def _upload_video(self, processed_video: ProcessedVideo) -> bool:
        """Upload a processed video to cloud storage."""
        try:
            # Try Google Drive first
            if self.google_drive_available and self.google_drive_handler:
                try:
                    self.google_drive_handler.upload_file(
                        str(processed_video.processed_path),
                        processed_video.processed_path.name
                    )
                    logger.info(f"Uploaded to Google Drive: {processed_video.processed_path}")
                    return True
                except Exception as e:
                    logger.warning(f"Google Drive upload failed: {e}")

            # Try Box as fallback
            if self.box_available and self.box_handler:
                try:
                    self.box_handler.upload_file(
                        str(processed_video.processed_path),
                        processed_video.processed_path.name
                    )
                    logger.info(f"Uploaded to Box: {processed_video.processed_path}")
                    return True
                except Exception as e:
                    logger.warning(f"Box upload failed: {e}")

            logger.error("No upload handlers available")
            return False

        except Exception as e:
            logger.error(f"Error uploading video {processed_video.processed_path}: {e}")
            return False


class EdgeApplicationStateMachine:
    """
    Main Edge Application with state machine architecture.

    Coordinates all edge device functionality through a state machine
    with supporting background threads for asynchronous operations.
    """

    def __init__(self, status_callback: Optional[Callable[[ApplicationState, str], None]] = None):
        """
        Initialize the Edge Application state machine.

        Parameters
        ----------
        status_callback : Callable[[ApplicationState, str], None], optional
            Callback function for state and status updates
        """
        self.status_callback = status_callback
        self.current_state = ApplicationState.INITIALIZING
        self.running = False

        # Core EdgeApplication for GoPro/YOLO functionality
        self.edge_app: Optional[EdgeApplication] = None

        # Background threads
        self.ble_thread: Optional[BLELoggerThread] = None
        self.processor_thread: Optional[PostProcessorThread] = None
        self.uploader_thread: Optional[UploaderThread] = None

        # Inter-thread communication
        self.video_queue = queue.Queue()
        self.processed_queue = queue.Queue()

        # State management
        self.state_lock = threading.Lock()
        self.shutdown_event = threading.Event()

        # Detection tracking
        self.last_detection_time = 0
        self.recording_start_time = 0
        self.recording_duration = 30.0  # Default recording duration in seconds

    def _log_status(self, message: str):
        """Log status with callback notification."""
        logger.info(f"[{self.current_state.value.upper()}] {message}")
        if self.status_callback:
            self.status_callback(self.current_state, message)

    def _change_state(self, new_state: ApplicationState):
        """Change application state with logging."""
        with self.state_lock:
            old_state = self.current_state
            self.current_state = new_state
            self._log_status(f"State transition: {old_state.value} -> {new_state.value}")

    def _thread_status_callback(self, event_type: str, data: Any):
        """Handle status updates from background threads."""
        logger.debug(f"Thread event: {event_type} - {data}")

        if event_type == "ble_data":
            # Handle BLE data
            self._log_status(f"BLE data received: {data.get('name', 'Unknown')}")

        elif event_type == "processing_started":
            self._log_status(f"Started processing: {data.file_path.name}")

        elif event_type == "processing_completed":
            self._log_status(f"Completed processing: {data.processed_path.name}")

        elif event_type == "upload_started":
            self._log_status(f"Started upload: {data.processed_path.name}")

        elif event_type == "upload_completed":
            self._log_status(f"Completed upload: {data.processed_path.name}")

    def _detection_callback(self, detection: DetectionResult):
        """Handle detection results from EdgeApplication."""
        self.last_detection_time = time.time()

        if self.current_state == ApplicationState.SEARCHING_FOR_WAKEBOARDER:
            self._log_status("Wakeboarder detected! Starting recording...")
            self._change_state(ApplicationState.RECORDING)
            self.recording_start_time = time.time()

            # Trigger recording in EdgeApplication
            if self.edge_app:
                self.edge_app.trigger_hindsight_clip()

    def initialize(self) -> bool:
        """Initialize all subsystems."""
        try:
            self._log_status("Initializing Edge Application...")

            # Initialize core EdgeApplication
            self.edge_app = EdgeApplication(
                detection_callback=self._detection_callback,
                log_callback=self._edge_log_callback
            )

            if not self.edge_app.initialize():
                self._log_status("Failed to initialize EdgeApplication")
                return False

            # Initialize background threads
            self.ble_thread = BLELoggerThread(status_callback=self._thread_status_callback)
            self.processor_thread = PostProcessorThread(
                self.video_queue,
                self.processed_queue,
                status_callback=self._thread_status_callback
            )
            self.uploader_thread = UploaderThread(
                self.processed_queue,
                status_callback=self._thread_status_callback
            )

            # Start background threads
            self.ble_thread.start_logging()
            self.processor_thread.start_processing()
            self.uploader_thread.start_uploading()

            self._log_status("All subsystems initialized successfully")
            return True

        except Exception as e:
            self._log_status(f"Initialization failed: {e}")
            return False

    def _edge_log_callback(self, level: str, message: str):
        """Handle log messages from EdgeApplication."""
        logger.info(f"EdgeApp [{level}]: {message}")

    def start_system(self) -> bool:
        """Start the complete Edge system."""
        if not self.initialize():
            return False

        try:
            # Start EdgeApplication system
            if not self.edge_app.start_system():
                self._log_status("Failed to start EdgeApplication system")
                return False

            self.running = True
            self._change_state(ApplicationState.SEARCHING_FOR_WAKEBOARDER)

            return True

        except Exception as e:
            self._log_status(f"Failed to start system: {e}")
            return False

    def run(self):
        """Main state machine loop."""
        if not self.start_system():
            return

        self._log_status("Edge Application state machine started")

        try:
            while self.running and not self.shutdown_event.is_set():
                current_time = time.time()

                if self.current_state == ApplicationState.SEARCHING_FOR_WAKEBOARDER:
                    # Continue searching for wakeboarder
                    time.sleep(0.1)

                elif self.current_state == ApplicationState.RECORDING:
                    # Check if recording duration has elapsed
                    if current_time - self.recording_start_time >= self.recording_duration:
                        self._log_status("Recording completed, returning to search mode")

                        # Create mock video file for processing
                        video_file = VideoFile(
                            file_path=Path(f"/tmp/recording_{int(self.recording_start_time)}.mp4"),
                            timestamp=self.recording_start_time,
                            metadata={"detection_time": self.last_detection_time}
                        )

                        # Queue for processing
                        self.video_queue.put(video_file)

                        self._change_state(ApplicationState.SEARCHING_FOR_WAKEBOARDER)
                    else:
                        time.sleep(0.1)

                elif self.current_state == ApplicationState.SHUTDOWN:
                    break

                else:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            self._log_status("Received interrupt signal")
        except Exception as e:
            self._log_status(f"State machine error: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """Gracefully shutdown all threads and systems."""
        self._log_status("Initiating graceful shutdown...")
        self._change_state(ApplicationState.SHUTDOWN)
        self.running = False
        self.shutdown_event.set()

        # Stop background threads
        if self.ble_thread:
            self.ble_thread.stop_logging()

        if self.processor_thread:
            self.processor_thread.stop_processing()
            # Send shutdown signal
            self.video_queue.put(None)

        if self.uploader_thread:
            self.uploader_thread.stop_uploading()
            # Send shutdown signal
            self.processed_queue.put(None)

        # Stop EdgeApplication
        if self.edge_app:
            self.edge_app.stop_system()

        # Wait for threads to finish
        threads = [self.ble_thread, self.processor_thread, self.uploader_thread]
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning(f"Thread {thread.name} did not shutdown gracefully")

        self._log_status("Shutdown complete")

    def get_state(self) -> ApplicationState:
        """Get current application state."""
        with self.state_lock:
            return self.current_state

    def is_running(self) -> bool:
        """Check if the state machine is running."""
        return self.running and not self.shutdown_event.is_set()


def main():
    """Main entry point for the Edge Application."""
    logging.basicConfig(level=logging.INFO)

    def status_callback(state: ApplicationState, message: str):
        """Status callback for demonstration."""
        print(f"[{state.value.upper()}] {message}")

    # Create and run the Edge Application
    app = EdgeApplicationStateMachine(status_callback=status_callback)

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()