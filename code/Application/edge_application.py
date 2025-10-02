"""
Edge Application with Modular Architecture

This is the refactored version of the Edge Application that uses a clean modular
architecture with specialized modules for different responsibilities.

The application now leverages:
- StatusManager for status and callback management
- StreamProcessor for video processing and YOLO detection
- EdgeSystemCoordinator for GoPro/system coordination
- EdgeStateMachine for high-level state management
- EdgeThreadManager for background thread management

This provides better maintainability, testability, and separation of concerns
compared to the original monolithic design.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import numpy as np

# Add module paths
MODULE_DIR = Path(__file__).resolve().parents[1] / "modules"
sys.path.append(str(MODULE_DIR))

# Import our new modular components
from StatusManager import StatusManager, SystemStatus, EdgeStatus, DetectionResult
from StreamProcessor import StreamProcessor
from EdgeSystemCoordinator import EdgeSystemCoordinator
from EdgeStateMachine import EdgeStateMachine, ApplicationState
from EdgeThreadManager import EdgeThreadManager
from EdgeApplicationConfig import EdgeApplicationConfig


logger = logging.getLogger(__name__)


class EdgeApplication:
    """
    Refactored Edge Application with modular architecture.

    This class now serves as a lightweight orchestration layer that coordinates
    specialized modules rather than implementing all functionality directly.
    It maintains backward compatibility with the original interface while
    providing much cleaner internal architecture.
    """

    def __init__(self,
                 status_callback: Optional[Callable[[SystemStatus], None]] = None,
                 detection_callback: Optional[Callable[[DetectionResult], None]] = None,
                 ble_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                 log_callback: Optional[Callable[[str, str], None]] = None,
                 frame_callback: Optional[Callable[[np.ndarray], None]] = None,
                 config: Optional[EdgeApplicationConfig] = None):
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
        config : EdgeApplicationConfig, optional
            Configuration object with Edge Application parameters
        """
        # Configuration
        self.config = config if config else EdgeApplicationConfig()

        # Initialize status manager with all callbacks
        self.status_manager = StatusManager(
            status_callback=status_callback,
            detection_callback=detection_callback,
            ble_callback=ble_callback,
            log_callback=log_callback,
            frame_callback=frame_callback
        )

        # Initialize system coordinator
        self.system_coordinator = EdgeSystemCoordinator(
            status_manager=self.status_manager,
            detection_callback=self._handle_detection,
            config=self.config
        )

        # State tracking
        self.initialized = False
        self.running = False

    def _handle_detection(self, detection: DetectionResult) -> None:
        """Handle detection results from system coordinator."""
        # Forward to status manager for callback handling
        self.status_manager.trigger_detection_callback(detection)

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
            self.status_manager.log("info", "Initializing Edge Application...")
            self.status_manager.update_status(overall_status=EdgeStatus.INITIALIZING)

            # Initialize system coordinator
            if not self.system_coordinator.initialize(config_path):
                self.status_manager.log("error", "Failed to initialize system coordinator")
                return False

            self.initialized = True
            self.status_manager.update_status(overall_status=EdgeStatus.READY)
            self.status_manager.log("info", "Edge Application initialized successfully")
            return True

        except Exception as e:
            self.status_manager.log("error", f"Failed to initialize Edge Application: {e}")
            self.status_manager.update_status(overall_status=EdgeStatus.ERROR)
            return False

    def connect_gopro(self) -> bool:
        """Connect to GoPro camera."""
        return self.system_coordinator.connect_gopro()

    def start_preview(self) -> bool:
        """Start GoPro preview stream."""
        return self.system_coordinator.start_preview()

    def stop_preview(self) -> bool:
        """Stop GoPro preview stream."""
        return self.system_coordinator.stop_preview()

    def start_ble_logging(self) -> bool:
        """Start BLE beacon logging in background thread."""
        return self.system_coordinator.start_ble_logging()

    def trigger_hindsight(self) -> bool:
        """Enable hindsight mode on the camera."""
        return self.system_coordinator.trigger_hindsight()

    def trigger_hindsight_clip(self) -> bool:
        """Trigger a hindsight clip recording."""
        return self.system_coordinator.trigger_hindsight_clip()

    def start_system(self) -> bool:
        """Start the complete Edge system."""
        if not self.initialized:
            self.status_manager.log("error", "System not initialized")
            return False

        try:
            self.status_manager.log("info", "Starting Edge system...")
            self.running = True

            # Start system coordinator
            if not self.system_coordinator.start_system():
                return False

            self.status_manager.log("info", "Edge system started successfully")
            return True

        except Exception as e:
            self.status_manager.log("error", f"Failed to start system: {e}")
            self.status_manager.update_status(overall_status=EdgeStatus.ERROR)
            return False

    def stop_system(self) -> None:
        """Stop the Edge system and cleanup resources."""
        self.status_manager.log("info", "Stopping Edge system...")
        self.running = False

        # Stop system coordinator
        if self.system_coordinator:
            self.system_coordinator.stop_system()

        # Update final status
        self.status_manager.update_status(overall_status=EdgeStatus.STOPPED)
        self.status_manager.log("info", "Edge system stopped")

    def get_status(self) -> SystemStatus:
        """Get current system status."""
        return self.status_manager.get_status()

    def is_running(self) -> bool:
        """Check if the system is running."""
        return self.running and self.initialized


class EdgeApplicationStateMachine:
    """
    Edge Application with state machine architecture.

    This is the high-level orchestrator that coordinates the EdgeApplication
    with the state machine for complete system management.
    """

    def __init__(self,
                 status_callback: Optional[Callable[[ApplicationState, str], None]] = None,
                 config: Optional[EdgeApplicationConfig] = None):
        """
        Initialize the Edge Application state machine.

        Parameters
        ----------
        status_callback : Callable[[ApplicationState, str], None], optional
            Callback function for state and status updates
        config : EdgeApplicationConfig, optional
            Configuration object with Edge Application parameters
        """
        # Configuration
        self.config = config if config else EdgeApplicationConfig()

        # Initialize Edge Application with config
        self.edge_app = EdgeApplication(config=self.config)

        # Initialize state machine with system coordinator and config
        self.state_machine = EdgeStateMachine(
            status_callback=status_callback,
            edge_system_coordinator=self.edge_app.system_coordinator,
            config=self.config
        )

    def run(self):
        """Run the state machine."""
        self.state_machine.run()

    def shutdown(self):
        """Shutdown the system."""
        self.state_machine.shutdown()
        self.edge_app.stop_system()

    def get_state(self) -> ApplicationState:
        """Get current application state."""
        return self.state_machine.get_state()

    def is_running(self) -> bool:
        """Check if the system is running."""
        return self.state_machine.is_running()

    def get_status(self) -> SystemStatus:
        """Get current system status."""
        return self.edge_app.get_status()

    def set_recording_duration(self, duration: float) -> None:
        """Set recording duration."""
        self.state_machine.set_recording_duration(duration)

    def get_system_stats(self) -> dict:
        """Get comprehensive system statistics."""
        stats = self.state_machine.get_system_stats()
        stats['edge_app_status'] = self.edge_app.get_status().__dict__
        return stats


def main():
    """Main entry point for the Edge Application."""
    logging.basicConfig(level=logging.INFO)

    def status_callback(state: ApplicationState, message: str):
        """Status callback for demonstration."""
        print(f"[{state.value.upper()}] {message}")

    # Create and run the Edge Application with state machine
    app = EdgeApplicationStateMachine(status_callback=status_callback)

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()