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


logger = logging.getLogger(__name__)


class ApplicationState(Enum):
    """Main application states for the state machine."""
    INITIALIZING = "initializing"
    SEARCHING_FOR_WAKEBOARDER = "searching_for_wakeboarder"
    RECORDING = "recording"
    SHUTDOWN = "shutdown"


class EdgeStateMachine:
    """
    Main Edge Application state machine.

    Coordinates all edge device functionality through a state machine
    with supporting background threads for asynchronous operations.
    """

    def __init__(self,
                 status_callback: Optional[Callable[[ApplicationState, str], None]] = None,
                 edge_system_coordinator=None):
        """
        Initialize the Edge State Machine.

        Parameters
        ----------
        status_callback : Callable[[ApplicationState, str], None], optional
            Callback function for state and status updates
        edge_system_coordinator : EdgeSystemCoordinator, optional
            System coordinator for managing GoPro/YOLO functionality
        """
        self.status_callback = status_callback
        self.edge_system_coordinator = edge_system_coordinator
        self.current_state = ApplicationState.INITIALIZING
        self.running = False

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
        self.recording_duration = 30.0  # Default recording duration in seconds

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
        self.last_detection_time = time.time()

        if self.current_state == ApplicationState.SEARCHING_FOR_WAKEBOARDER:
            self._log_status("Wakeboarder detected! Starting recording...")
            self._change_state(ApplicationState.RECORDING)
            self.recording_start_time = time.time()

            # Trigger recording in EdgeSystemCoordinator
            if self.edge_system_coordinator:
                self.edge_system_coordinator.trigger_hindsight_clip()

    def initialize(self) -> bool:
        """Initialize all subsystems."""
        try:
            self._log_status("Initializing Edge Application State Machine...")

            # Initialize background thread manager
            self.thread_manager = EdgeThreadManager(status_callback=self._thread_status_callback)

            # Initialize EdgeSystemCoordinator if provided
            if self.edge_system_coordinator:
                if not self.edge_system_coordinator.initialize():
                    self._log_status("Failed to initialize EdgeSystemCoordinator")
                    return False

                # Set up detection callback
                self.edge_system_coordinator.set_detection_callback(self._detection_callback)

            # Start background threads
            if not self.thread_manager.start_all_threads():
                self._log_status("Failed to start background threads")
                return False

            self._log_status("All subsystems initialized successfully")
            return True

        except Exception as e:
            self._log_status(f"Initialization failed: {e}")
            return False

    def start_system(self) -> bool:
        """Start the complete Edge system."""
        if not self.initialize():
            return False

        try:
            # Start EdgeSystemCoordinator if available
            if self.edge_system_coordinator:
                if not self.edge_system_coordinator.start_system():
                    self._log_status("Failed to start EdgeSystemCoordinator")
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

                        # Create video file entry for processing
                        video_file = VideoFile(
                            file_path=Path(f"/tmp/recording_{int(self.recording_start_time)}.mp4"),
                            timestamp=self.recording_start_time,
                            metadata={"detection_time": self.last_detection_time}
                        )

                        # Queue for processing
                        if self.thread_manager:
                            self.thread_manager.queue_video_for_processing(video_file)

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

        # Stop EdgeSystemCoordinator
        if self.edge_system_coordinator:
            self.edge_system_coordinator.stop_system()

        # Stop background threads
        if self.thread_manager:
            self.thread_manager.stop_all_threads()

        self._log_status("Shutdown complete")

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