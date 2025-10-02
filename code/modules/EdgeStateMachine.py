"""
Edge State Machine Module

This module provides the main state machine architecture for the Edge Application system.
It coordinates the high-level application states and manages transitions between different
operational modes.
"""

import logging
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Any

from StatusManager import StatusManager, DetectionResult
from EdgeThreadManager import EdgeThreadManager, VideoFile
from EdgeApplicationConfig import EdgeApplicationConfig


logger = logging.getLogger(__name__)


class ApplicationState(Enum):
    """Main application states for the state machine."""
    INITIALIZE = "initialize"
    LOOKING_FOR_WAKEBOARDER = "looking_for_wakeboarder"
    RECORDING = "recording"
    ERROR = "error"
    STOPPING = "stopping"


class EdgeStateMachine:
    """
    Main Edge Application state machine.

    Coordinates all edge device functionality through a state machine
    with supporting background threads for asynchronous operations.
    """

    def __init__(self,
                 status_callback: Optional[Callable[[ApplicationState, str], None]] = None,
                 edge_system_coordinator=None,
                 config: Optional[EdgeApplicationConfig] = None):
        """
        Initialize the Edge State Machine.

        Parameters
        ----------
        status_callback : Callable[[ApplicationState, str], None], optional
            Callback function for state and status updates
        edge_system_coordinator : EdgeSystemCoordinator, optional
            System coordinator for managing GoPro/YOLO functionality
        config : EdgeApplicationConfig, optional
            Configuration object with Edge Application parameters
        """
        self.status_callback = status_callback
        self.edge_system_coordinator = edge_system_coordinator
        self.current_state = ApplicationState.INITIALIZE
        self.running = False

        # Configuration
        self.config = config if config else EdgeApplicationConfig()

        # Status manager for internal logging
        self.status_manager = StatusManager(log_callback=self._internal_log_callback)

        # Background thread manager
        self.thread_manager: Optional[EdgeThreadManager] = None

        # State management
        self.state_lock = threading.Lock()
        self.shutdown_event = threading.Event()

        # Detection and recording tracking
        self.last_detection_time = 0
        self.recording_start_time = 0
        self.recording_duration = self.config.get_recording_duration()
        self.detection_cooldown = self.config.get_detection_cooldown()
        self.last_valid_detection_time = 0

        # Error tracking
        self.error_message = ""
        self.error_count = 0
        self.max_error_restarts = self.config.get_max_error_restarts()
        self.error_restart_delay = self.config.get_error_restart_delay()

        # State-specific flags
        self._hindsight_enabled = False

    def _internal_log_callback(self, level: str, message: str) -> None:
        """Handle internal log messages."""
        logger.info(f"StateMachine [{level}]: {message}")

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
        """Handle detection results from EdgeSystemCoordinator."""
        current_time = time.time()
        self.last_detection_time = current_time

        if self.current_state == ApplicationState.LOOKING_FOR_WAKEBOARDER:
            # Check detection cooldown
            if (current_time - self.last_valid_detection_time) < self.detection_cooldown:
                logger.debug(f"Detection ignored due to cooldown "
                           f"({current_time - self.last_valid_detection_time:.2f}s < {self.detection_cooldown}s)")
                return

            self.last_valid_detection_time = current_time
            self._log_status("Wakeboarder detected! Starting recording...")
            self._change_state(ApplicationState.RECORDING)
            self.recording_start_time = current_time

            # Trigger recording in EdgeSystemCoordinator
            if self.edge_system_coordinator:
                self.edge_system_coordinator.trigger_hindsight_clip()

    def _execute_initialize_state(self) -> bool:
        """
        Execute initialization tasks as per state machine design.

        This includes:
        - Connect to GoPro
        - Start GoPro preview
        - Start Bluetooth logging thread
        - Start GUI thread (if applicable)
        - Start Cloud upload thread
        - Start Post-processing thread

        Returns
        -------
        bool
            True if all initialization successful, transitions to LOOKING_FOR_WAKEBOARDER
            False if initialization failed, transitions to ERROR
        """
        try:
            self._log_status("Executing initialization tasks...")

            # Initialize background thread manager
            self.thread_manager = EdgeThreadManager(status_callback=self._thread_status_callback)

            # Initialize EdgeSystemCoordinator if provided
            if self.edge_system_coordinator:
                if not self.edge_system_coordinator.initialize():
                    self.error_message = "Failed to initialize EdgeSystemCoordinator"
                    self._log_status(self.error_message)
                    return False

                # Set up detection callback
                self.edge_system_coordinator.set_detection_callback(self._detection_callback)

                # Connect to GoPro
                self._log_status("Connecting to GoPro...")
                if not self.edge_system_coordinator.connect_gopro():
                    self.error_message = "Failed to connect to GoPro"
                    self._log_status(self.error_message)
                    return False

                # Start GoPro preview
                self._log_status("Starting GoPro preview...")
                if not self.edge_system_coordinator.start_preview():
                    self.error_message = "Failed to start GoPro preview"
                    self._log_status(self.error_message)
                    return False

                # Start BLE logging thread
                self._log_status("Starting Bluetooth logging thread...")
                if not self.edge_system_coordinator.start_ble_logging():
                    self.error_message = "Failed to start BLE logging"
                    self._log_status(self.error_message)
                    return False

            # Start background threads (GUI, Cloud upload, Post-processing)
            self._log_status("Starting background threads...")
            if not self.thread_manager.start_all_threads():
                self.error_message = "Failed to start background threads"
                self._log_status(self.error_message)
                return False

            self._log_status("Initialization complete")
            return True

        except Exception as e:
            self.error_message = f"Initialization failed: {e}"
            self._log_status(self.error_message)
            return False

    def _execute_looking_for_wakeboarder_state(self) -> None:
        """
        Execute LookingForWakeboarder state behavior.

        This includes:
        - Enable Hindsight on GoPro (if configured)
        - Start YOLO detection loop (if enabled)
        - Wait for wakeboarder detection (handled by detection callback)
        """
        # Enable hindsight mode on entry (only once)
        if hasattr(self, '_hindsight_enabled') and self._hindsight_enabled:
            return

        # Check if hindsight mode is enabled in config
        if self.config.get_hindsight_mode_enabled():
            self._log_status("Enabling Hindsight mode on GoPro...")
            if self.edge_system_coordinator:
                if self.edge_system_coordinator.trigger_hindsight():
                    self._hindsight_enabled = True
                    self._log_status("Hindsight mode enabled")
                else:
                    self.error_message = "Failed to enable hindsight mode"
                    self._change_state(ApplicationState.ERROR)
                    return
        else:
            self._log_status("Hindsight mode disabled by configuration")
            self._hindsight_enabled = True  # Mark as handled

        # Check if YOLO detection is enabled
        if self.config.get_yolo_enabled():
            self._log_status("Starting YOLO detection loop")
            # Start stream processing (YOLO detection)
            if self.edge_system_coordinator and \
               hasattr(self.edge_system_coordinator, 'stream_processor') and \
               self.edge_system_coordinator.stream_processor:
                if not self.edge_system_coordinator.stream_processor.start_processing():
                    self.error_message = "Failed to start YOLO detection"
                    self._change_state(ApplicationState.ERROR)
                    return
            self._log_status("YOLO detection active - Waiting for wakeboarder...")
        else:
            self._log_status("WARNING: YOLO detection is DISABLED by configuration")
            self._log_status("State machine will remain in LOOKING_FOR_WAKEBOARDER state")
            self._log_status("No automatic wakeboarder detection will occur")

    def _execute_recording_state(self, current_time: float) -> None:
        """
        Execute Recording state behavior.

        This includes:
        - Trigger recording (done in transition)
        - Wait for recording duration
        - Send clip to post-processing thread
        - Transition back to LookingForWakeboarder
        """
        # Check if recording duration has elapsed
        if current_time - self.recording_start_time >= self.recording_duration:
            self._log_status("Recording completed, sending clip to post-processing")

            # Create video file entry for processing
            video_file = VideoFile(
                file_path=Path(f"/tmp/recording_{int(self.recording_start_time)}.mp4"),
                timestamp=self.recording_start_time,
                metadata={"detection_time": self.last_detection_time}
            )

            # Send clip to post-processing thread
            if self.thread_manager:
                self.thread_manager.queue_video_for_processing(video_file)
                self._log_status("Clip sent to post-processing thread")

            # Reset hindsight flag so it can be re-enabled in next cycle
            self._hindsight_enabled = False

            # Transition back to looking for wakeboarder
            self._change_state(ApplicationState.LOOKING_FOR_WAKEBOARDER)

    def _execute_error_state(self) -> None:
        """
        Execute Error state behavior.

        This includes:
        - Display/log error
        - Attempt restart if under max restart count
        - Otherwise, transition to STOPPING
        """
        self._log_status(f"ERROR: {self.error_message}")

        # Check if we should attempt restart
        if self.error_count < self.max_error_restarts:
            self.error_count += 1
            self._log_status(f"Attempting restart ({self.error_count}/{self.max_error_restarts})...")

            # Cleanup before restart
            if self.edge_system_coordinator:
                self.edge_system_coordinator.stop_system()

            # Wait before restarting (use configured delay)
            self._log_status(f"Waiting {self.error_restart_delay}s before restart...")
            time.sleep(self.error_restart_delay)

            # Transition back to INITIALIZE for restart
            self._change_state(ApplicationState.INITIALIZE)
            self._hindsight_enabled = False  # Reset hindsight flag
        else:
            self._log_status("Max restart attempts reached, shutting down")
            self._change_state(ApplicationState.STOPPING)

    def _execute_stopping_state(self) -> None:
        """
        Execute Stopping state behavior.

        This includes:
        - Close all threads
        - Cleanup resources
        - Final shutdown
        """
        self._log_status("Executing graceful shutdown...")

        # Stop EdgeSystemCoordinator
        if self.edge_system_coordinator:
            self.edge_system_coordinator.stop_system()

        # Stop background threads
        if self.thread_manager:
            self.thread_manager.stop_all_threads()

        self.running = False
        self._log_status("Shutdown complete")

    def run(self):
        """Main state machine loop."""
        self.running = True
        self._log_status("Edge Application state machine started")

        try:
            while self.running and not self.shutdown_event.is_set():
                current_time = time.time()

                if self.current_state == ApplicationState.INITIALIZE:
                    # Execute initialization
                    if self._execute_initialize_state():
                        # Initialization successful, transition to looking for wakeboarder
                        self._change_state(ApplicationState.LOOKING_FOR_WAKEBOARDER)
                    else:
                        # Initialization failed, transition to error
                        self._change_state(ApplicationState.ERROR)

                elif self.current_state == ApplicationState.LOOKING_FOR_WAKEBOARDER:
                    # Execute looking for wakeboarder state
                    self._execute_looking_for_wakeboarder_state()
                    # Continue searching (detection happens via callback)
                    time.sleep(0.1)

                elif self.current_state == ApplicationState.RECORDING:
                    # Execute recording state
                    self._execute_recording_state(current_time)
                    time.sleep(0.1)

                elif self.current_state == ApplicationState.ERROR:
                    # Execute error state
                    self._execute_error_state()
                    # Give some time for restart preparation
                    time.sleep(0.5)

                elif self.current_state == ApplicationState.STOPPING:
                    # Execute stopping state
                    self._execute_stopping_state()
                    break

                else:
                    self._log_status(f"Unknown state: {self.current_state}")
                    time.sleep(0.1)

        except KeyboardInterrupt:
            self._log_status("Received interrupt signal")
            self._change_state(ApplicationState.STOPPING)
            self._execute_stopping_state()
        except Exception as e:
            self._log_status(f"State machine error: {e}")
            self.error_message = str(e)
            self._change_state(ApplicationState.ERROR)
            self._execute_error_state()

    def shutdown(self):
        """Gracefully shutdown all threads and systems."""
        self._log_status("Initiating graceful shutdown...")
        self._change_state(ApplicationState.STOPPING)
        self.shutdown_event.set()
        # The actual shutdown will be handled by STOPPING state

    def get_state(self) -> ApplicationState:
        """Get current application state."""
        with self.state_lock:
            return self.current_state

    def is_running(self) -> bool:
        """Check if the state machine is running."""
        return self.running and not self.shutdown_event.is_set()

    def set_recording_duration(self, duration: float) -> None:
        """Set the recording duration in seconds."""
        self.recording_duration = duration
        self._log_status(f"Recording duration set to {duration} seconds")

    def force_state_transition(self, new_state: ApplicationState) -> None:
        """Force a state transition (for testing/debugging)."""
        self._log_status(f"Forcing state transition to {new_state.value}")
        self._change_state(new_state)

    def get_system_stats(self) -> dict:
        """Get current system statistics."""
        stats = {
            'current_state': self.current_state.value,
            'running': self.running,
            'last_detection_time': self.last_detection_time,
            'recording_start_time': self.recording_start_time,
            'recording_duration': self.recording_duration
        }

        # Add thread manager stats if available
        if self.thread_manager:
            stats['queue_sizes'] = self.thread_manager.get_queue_sizes()

        return stats