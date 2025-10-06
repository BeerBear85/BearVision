"""
Edge System Coordinator Module

This module provides focused coordination of core Edge Application components including
GoPro camera management, YOLO detection, BLE beacon handling, and system lifecycle management.
This replaces the monolithic EdgeApplication class with a cleaner, more maintainable design.
"""

import logging
import threading
import asyncio
import time
from pathlib import Path
from typing import Optional, Callable

from StatusManager import StatusManager, EdgeStatus, DetectionResult
from StreamProcessor import StreamProcessor
from GoProController import GoProController
from DnnHandler import DnnHandler
from ble_beacon_handler import BleBeaconHandler
from ConfigurationHandler import ConfigurationHandler
from EdgeApplicationConfig import EdgeApplicationConfig


logger = logging.getLogger(__name__)


class EdgeSystemCoordinator:
    """
    Coordinates core Edge Application functionality.

    This class serves as a focused coordinator for:
    - GoPro camera connection and management
    - YOLO person detection system
    - BLE beacon monitoring
    - Stream processing and frame handling
    - System lifecycle management

    It replaces the monolithic EdgeApplication class with a cleaner design
    that leverages specialized modules for specific functionality.
    """

    def __init__(self,
                 status_manager: Optional[StatusManager] = None,
                 detection_callback: Optional[Callable[[DetectionResult], None]] = None,
                 config: Optional[EdgeApplicationConfig] = None):
        """
        Initialize the Edge System Coordinator.

        Parameters
        ----------
        status_manager : StatusManager, optional
            Status manager for logging and status updates
        detection_callback : Callable[[DetectionResult], None], optional
            Callback function for detection results
        config : EdgeApplicationConfig, optional
            Configuration object with Edge Application parameters
        """
        # Status management
        if status_manager:
            self.status_manager = status_manager
        else:
            self.status_manager = StatusManager()

        # Detection callback
        self.detection_callback = detection_callback

        # Configuration
        self.config = config if config else EdgeApplicationConfig()

        # Core system components
        self.gopro_controller: Optional[GoProController] = None
        self.dnn_handler: Optional[DnnHandler] = None
        self.ble_handler: Optional[BleBeaconHandler] = None
        self.stream_processor: Optional[StreamProcessor] = None

        # Threading components
        self.ble_thread: Optional[threading.Thread] = None

        # System state
        self.initialized = False
        self.running = False

        # Configuration tracking
        self.config_loaded = False

    def set_detection_callback(self, callback: Callable[[DetectionResult], None]) -> None:
        """Set the detection callback function."""
        self.detection_callback = callback

    def initialize(self, config_path: Optional[str] = None) -> bool:
        """
        Initialize the Edge System.

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
            self.status_manager.log("info", "Initializing Edge System Coordinator...")
            self.status_manager.update_status(overall_status=EdgeStatus.INITIALIZING)

            # Load configuration
            if not self._load_config(config_path):
                return False

            # Initialize YOLO detection
            if not self._initialize_yolo():
                return False

            # Initialize BLE handler
            if not self._initialize_ble():
                return False

            # Initialize stream processor
            self._initialize_stream_processor()

            self.initialized = True
            self.status_manager.update_status(overall_status=EdgeStatus.READY)
            self.status_manager.log("info", "Edge System Coordinator initialized successfully")
            return True

        except Exception as e:
            self.status_manager.log("error", f"Failed to initialize Edge System Coordinator: {e}")
            self.status_manager.update_status(overall_status=EdgeStatus.ERROR)
            return False

    def _load_config(self, config_path: Optional[str] = None) -> bool:
        """Load configuration file."""
        try:
            if config_path is None:
                config_path = Path(__file__).resolve().parents[2] / "config.ini"

            ConfigurationHandler.read_config_file(str(config_path))
            self.config_loaded = True
            self.status_manager.log("info", f"Configuration loaded from {config_path}")
            return True

        except Exception as e:
            self.status_manager.log("error", f"Failed to load configuration: {e}")
            return False

    def _initialize_yolo(self) -> bool:
        """Initialize YOLO detection system (if enabled in config)."""
        # Check if YOLO is enabled in configuration
        if not self.config.get_yolo_enabled():
            self.status_manager.log("info", "YOLO detection disabled by configuration")
            return True  # Return true as this is expected behavior

        try:
            yolo_model = self.config.get_yolo_model()
            self.dnn_handler = DnnHandler(yolo_model)
            self.dnn_handler.init()
            self.status_manager.update_status(yolo_active=True)
            self.status_manager.log("info", f"YOLO {yolo_model} model initialized")
            return True

        except Exception as e:
            self.status_manager.log("error", f"Failed to initialize YOLO: {e}")
            return False

    def _initialize_ble(self) -> bool:
        """Initialize BLE beacon handler."""
        try:
            self.ble_handler = BleBeaconHandler()
            self.status_manager.log("info", "BLE beacon handler initialized")
            return True

        except Exception as e:
            self.status_manager.log("error", f"Failed to initialize BLE: {e}")
            return False

    def _initialize_stream_processor(self) -> None:
        """Initialize stream processor with detection callback."""
        self.stream_processor = StreamProcessor(
            status_manager=self.status_manager,
            dnn_handler=self.dnn_handler,
            config=self.config
        )

        # Set up detection callback chain
        if self.detection_callback:
            original_callback = self.stream_processor.status_manager.detection_callback

            def combined_callback(detection: DetectionResult):
                # Call original callback if it exists
                if original_callback:
                    original_callback(detection)
                # Call our callback
                self.detection_callback(detection)

            self.stream_processor.status_manager.detection_callback = combined_callback

        self.status_manager.log("info", "Stream processor initialized")

    def connect_gopro(self) -> bool:
        """
        Connect to GoPro camera.

        Returns
        -------
        bool
            True if connection successful, False otherwise
        """
        try:
            self.status_manager.log("info", "Connecting to GoPro...")
            self.gopro_controller = GoProController()
            self.gopro_controller.connect()

            # Wait for connection to be fully established
            self.status_manager.log("info", "Verifying GoPro connection...")
            if not self._wait_for_gopro_ready():
                self.status_manager.log("error", "GoPro connection not ready after waiting")
                return False

            self.status_manager.log("info", "Configuring GoPro settings...")
            self.gopro_controller.configure()

            self.status_manager.update_status(gopro_connected=True)
            self.status_manager.log("info", "GoPro connected and configured successfully")
            return True

        except Exception as e:
            self.status_manager.log("error", f"Failed to connect to GoPro: {e}")
            self.status_manager.update_status(gopro_connected=False)
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
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                # Check GoPro connection status
                gopro = self.gopro_controller._gopro

                # Basic connection check
                if not hasattr(gopro, '_serial') or not gopro._serial:
                    self.status_manager.log("debug", f"GoPro serial not available (attempt {attempt + 1})")
                    time.sleep(delay)
                    delay = min(delay * 1.5, 3.0)
                    continue

                # Try to get camera status to verify connection
                try:
                    status = self.gopro_controller._run_in_thread(gopro.http_command.get_camera_state())
                    if status and hasattr(status, 'data') and status.data:
                        self.status_manager.log("info", f"GoPro connection ready (attempt {attempt + 1})")
                        return True
                    else:
                        self.status_manager.log("debug", f"GoPro status check returned empty data (attempt {attempt + 1})")
                except Exception as status_error:
                    self.status_manager.log("debug", f"GoPro status check failed: {status_error} (attempt {attempt + 1})")

                time.sleep(delay)
                delay = min(delay * 1.5, 3.0)

            except Exception as e:
                self.status_manager.log("debug", f"Error checking GoPro readiness: {e}")
                time.sleep(delay)
                delay = min(delay * 1.5, 3.0)

        self.status_manager.log("error", f"GoPro connection not ready after {max_retries} attempts")
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
            self.status_manager.log("error", "GoPro not connected")
            return False

        try:
            # Start preview and get the stream URL
            preview_url = self.gopro_controller.start_preview()
            self.status_manager.update_status(preview_active=True)
            self.status_manager.log("info", f"GoPro preview started, URL: {preview_url}")

            # Configure stream processor with the preview URL
            if self.stream_processor:
                self.stream_processor.set_preview_stream_url(preview_url)

            return True

        except Exception as e:
            self.status_manager.log("error", f"Failed to start preview: {e}")
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
            self.status_manager.update_status(preview_active=False)
            self.status_manager.log("info", "GoPro preview stopped")
            return True

        except Exception as e:
            self.status_manager.log("error", f"Failed to stop preview: {e}")
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
            self.status_manager.log("error", "BLE handler not initialized")
            return False

        try:
            self.ble_thread = threading.Thread(
                target=self._ble_worker,
                name="ble_worker",
                daemon=True
            )
            self.ble_thread.start()

            self.status_manager.update_status(ble_scanning=True)
            self.status_manager.log("info", "BLE beacon logging started")
            return True

        except Exception as e:
            self.status_manager.log("error", f"Failed to start BLE logging: {e}")
            return False

    def _ble_worker(self) -> None:
        """BLE worker thread function."""
        try:
            # Wrap the BLE handler's async function for the thread
            asyncio.run(self._ble_scan_loop())
        except Exception as e:
            self.status_manager.log("error", f"BLE worker error: {e}")
            self.status_manager.update_status(ble_scanning=False)

    async def _ble_scan_loop(self) -> None:
        """Async BLE scanning loop."""
        async def custom_process():
            while self.running:
                try:
                    advertisement = await self.ble_handler.advertisement_queue.get()
                    # Trigger BLE callback if available
                    self.status_manager.trigger_ble_callback(advertisement)
                    self.ble_handler.advertisement_queue.task_done()
                except Exception as e:
                    self.status_manager.log("error", f"BLE processing error: {e}")

        # Override the process method temporarily
        self.ble_handler.process_advertisements = custom_process

        # Start the scanning
        await self.ble_handler.start_scan_async(timeout=0.0)

    def trigger_hindsight(self) -> bool:
        """
        Enable hindsight mode on the camera.

        Returns
        -------
        bool
            True if hindsight mode enabled successfully, False otherwise
        """
        if not self.gopro_controller:
            self.status_manager.log("error", "GoPro not connected")
            return False

        try:
            success = self.gopro_controller.startHindsightMode()
            if success:
                self.status_manager.update_status(hindsight_mode=True)
                self.status_manager.log("info", "Hindsight mode enabled")
                return True
            else:
                self.status_manager.log("warning", "Failed to enable hindsight mode")
                return False

        except Exception as e:
            self.status_manager.log("error", f"Failed to enable hindsight mode: {e}")
            return False

    def trigger_hindsight_clip(self) -> bool:
        """
        Trigger a hindsight clip recording.

        Returns
        -------
        bool
            True if hindsight clip triggered successfully, False otherwise
        """
        if not self.gopro_controller:
            self.status_manager.log("error", "GoPro not connected")
            return False

        try:
            self.gopro_controller.start_hindsight_clip()
            self.status_manager.update_status(overall_status=EdgeStatus.RECORDING)
            self.status_manager.log("info", "Hindsight clip triggered")

            # Reset to looking for wakeboarder after a delay
            threading.Timer(5.0, self._reset_to_looking).start()

            return True

        except Exception as e:
            self.status_manager.log("error", f"Failed to trigger hindsight clip: {e}")
            return False

    def _reset_to_looking(self) -> None:
        """Reset status back to looking for wakeboarder."""
        self.status_manager.update_status(
            hindsight_mode=False,
            overall_status=EdgeStatus.LOOKING_FOR_WAKEBOARDER
        )
        self.status_manager.log("info", "Ready - Looking for wakeboarder")

    def start_system(self) -> bool:
        """
        Start the complete Edge system (basic components only).

        NOTE: This method has been simplified for state machine control.
        The state machine now controls:
        - When to enable hindsight mode (in LOOKING_FOR_WAKEBOARDER state)
        - When to start YOLO detection (in LOOKING_FOR_WAKEBOARDER state)

        This method only handles:
        - Connecting to GoPro
        - Starting preview
        - Starting BLE logging

        Returns
        -------
        bool
            True if all systems started successfully, False otherwise
        """
        if not self.initialized:
            self.status_manager.log("error", "System not initialized")
            return False

        try:
            self.status_manager.log("info", "Starting Edge system coordinator...")
            self.running = True

            # Note: GoPro connection, preview, and BLE logging are now
            # handled by the state machine during INITIALIZE state
            self.status_manager.log("info", "Edge system coordinator ready")

            return True

        except Exception as e:
            self.status_manager.log("error", f"Failed to start system: {e}")
            self.status_manager.update_status(overall_status=EdgeStatus.ERROR)
            return False

    def stop_system(self) -> None:
        """Stop the Edge system and cleanup resources."""
        self.status_manager.log("info", "Stopping Edge System Coordinator...")
        self.running = False

        # Stop stream processing
        if self.stream_processor:
            self.stream_processor.stop_processing()

        # Stop preview
        self.stop_preview()

        # Disconnect GoPro
        if self.gopro_controller:
            try:
                self.gopro_controller.disconnect()
                self.status_manager.log("info", "GoPro disconnected")
            except Exception as e:
                self.status_manager.log("warning", f"Error disconnecting GoPro: {e}")

        # Update status
        self.status_manager.update_status(
            overall_status=EdgeStatus.STOPPED,
            gopro_connected=False,
            preview_active=False,
            ble_scanning=False,
            hindsight_mode=False,
            recording=False
        )

        self.status_manager.log("info", "Edge System Coordinator stopped")

    def get_status(self):
        """Get current system status."""
        return self.status_manager.get_status()

    def is_running(self) -> bool:
        """Check if the system is running."""
        return self.running and self.initialized