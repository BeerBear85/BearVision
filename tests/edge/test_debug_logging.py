"""
Test for debug logging of state transitions in EdgeStateMachine.

This test verifies that all state transitions are logged at DEBUG level
with the required format: SOURCE_STATE → TARGET_STATE | TRIGGER
"""

import sys
import logging
import time
from pathlib import Path
from unittest import mock
from unittest.mock import Mock, patch
import pytest

# Add module paths for imports
MODULE_DIR = Path(__file__).resolve().parents[2] / 'code' / 'modules'
APP_DIR = Path(__file__).resolve().parents[2] / 'code' / 'Application'
sys.path.append(str(MODULE_DIR))
sys.path.append(str(APP_DIR))

from EdgeStateMachine import EdgeStateMachine, ApplicationState
from EdgeApplicationConfig import EdgeApplicationConfig


class TestEdgeStateMachineDebugLogging:
    """Test debug logging functionality for state transitions."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Mock configuration
        self.mock_config = Mock(spec=EdgeApplicationConfig)
        self.mock_config.get_recording_duration.return_value = 5.0
        self.mock_config.get_detection_cooldown.return_value = 1.0
        self.mock_config.get_max_error_restarts.return_value = 2
        self.mock_config.get_error_restart_delay.return_value = 0.1
        self.mock_config.get_hindsight_mode_enabled.return_value = True
        self.mock_config.get_yolo_enabled.return_value = True

        # Mock EdgeSystemCoordinator
        self.mock_coordinator = Mock()
        self.mock_coordinator.initialize.return_value = True
        self.mock_coordinator.connect_gopro.return_value = True
        self.mock_coordinator.start_preview.return_value = True
        self.mock_coordinator.start_ble_logging.return_value = True
        self.mock_coordinator.trigger_hindsight.return_value = True
        self.mock_coordinator.trigger_hindsight_clip.return_value = True
        self.mock_coordinator.stream_processor = Mock()
        self.mock_coordinator.stream_processor.start_processing.return_value = True

        # Create state machine
        self.state_machine = EdgeStateMachine(
            edge_system_coordinator=self.mock_coordinator,
            config=self.mock_config
        )

    def teardown_method(self):
        """Clean up after each test method."""
        if hasattr(self.state_machine, 'running') and self.state_machine.running:
            self.state_machine.shutdown()
            time.sleep(0.1)

    def test_debug_logging_format_consistency(self):
        """Test that state transitions produce debug logs with consistent format."""
        # Capture log records at DEBUG level
        with patch('EdgeStateMachine.logger') as mock_logger:
            # Test various state transitions with different triggers
            test_transitions = [
                (ApplicationState.LOOKING_FOR_WAKEBOARDER, "wakeboarder detected"),
                (ApplicationState.ERROR, "initialization failed"),
                (ApplicationState.STOPPING, "graceful shutdown requested"),
                (ApplicationState.INITIALIZE, "error restart attempt"),
            ]

            for target_state, trigger in test_transitions:
                self.state_machine._change_state(target_state, trigger)

                # Verify debug log was called with correct format
                debug_calls = mock_logger.debug.call_args_list
                assert len(debug_calls) > 0, f"No debug log for transition to {target_state.value}"

                # Check the most recent debug call
                last_debug_call = debug_calls[-1]
                log_message = last_debug_call[0][0]

                # Verify format: SOURCE_STATE → TARGET_STATE | TRIGGER
                expected_format = f"STATE TRANSITION: {self.state_machine.current_state.value} → {target_state.value} | {trigger}"
                # Note: current_state is updated during _change_state, so we check it includes the target
                assert "STATE TRANSITION:" in log_message, f"Debug log missing 'STATE TRANSITION:' prefix: {log_message}"
                assert "→" in log_message, f"Debug log missing arrow character: {log_message}"
                assert "|" in log_message, f"Debug log missing trigger separator: {log_message}"
                assert trigger in log_message, f"Debug log missing trigger '{trigger}': {log_message}"
                assert target_state.value in log_message, f"Debug log missing target state '{target_state.value}': {log_message}"

    def test_all_transition_triggers_logged(self):
        """Test that all state transition triggers are logged correctly."""
        with patch('EdgeStateMachine.logger') as mock_logger:
            # Test all trigger types from the implementation
            trigger_tests = [
                (ApplicationState.RECORDING, "wakeboarder detected"),
                (ApplicationState.ERROR, "hindsight mode enable failed"),
                (ApplicationState.ERROR, "YOLO detection start failed"),
                (ApplicationState.LOOKING_FOR_WAKEBOARDER, "recording duration elapsed"),
                (ApplicationState.INITIALIZE, "error restart attempt"),
                (ApplicationState.STOPPING, "max restart attempts reached"),
                (ApplicationState.LOOKING_FOR_WAKEBOARDER, "initialization successful"),
                (ApplicationState.ERROR, "initialization failed"),
                (ApplicationState.STOPPING, "keyboard interrupt received"),
                (ApplicationState.ERROR, "state machine exception"),
                (ApplicationState.STOPPING, "graceful shutdown requested"),
                (ApplicationState.RECORDING, "forced transition"),
            ]

            for target_state, trigger in trigger_tests:
                # Reset mock to isolate each test
                mock_logger.reset_mock()

                # Execute state change
                self.state_machine._change_state(target_state, trigger)

                # Verify debug log was called
                assert mock_logger.debug.called, f"No debug log for trigger: {trigger}"

                # Get the debug log message
                debug_call = mock_logger.debug.call_args[0][0]

                # Verify trigger is in the log message
                assert trigger in debug_call, f"Trigger '{trigger}' not found in debug log: {debug_call}"

    def test_default_trigger_when_none_provided(self):
        """Test that 'unknown' trigger is used when none provided."""
        with patch('EdgeStateMachine.logger') as mock_logger:
            # Call _change_state without trigger parameter
            self.state_machine._change_state(ApplicationState.ERROR)

            # Verify debug log was called with 'unknown' trigger
            assert mock_logger.debug.called
            debug_call = mock_logger.debug.call_args[0][0]
            assert "unknown" in debug_call, f"Default trigger 'unknown' not found: {debug_call}"

    def test_state_transition_filtering_by_keyword(self):
        """Test that debug logs can be filtered by 'STATE TRANSITION' keyword."""
        with patch('EdgeStateMachine.logger') as mock_logger:
            # Perform multiple state transitions
            self.state_machine._change_state(ApplicationState.RECORDING, "test trigger 1")
            self.state_machine._change_state(ApplicationState.ERROR, "test trigger 2")
            self.state_machine._change_state(ApplicationState.STOPPING, "test trigger 3")

            # Verify all debug calls contain the filterable keyword
            debug_calls = mock_logger.debug.call_args_list
            assert len(debug_calls) >= 3, "Expected at least 3 debug calls"

            for call in debug_calls:
                log_message = call[0][0]
                assert "STATE TRANSITION:" in log_message, f"Log not filterable by keyword: {log_message}"

    def test_debug_logging_preserves_thread_safety(self):
        """Test that debug logging doesn't affect thread safety of state changes."""
        import threading

        # Track any exceptions during concurrent access
        exceptions = []

        def change_state_worker(target_state, trigger):
            try:
                for i in range(5):
                    self.state_machine._change_state(target_state, f"{trigger}_{i}")
                    time.sleep(0.001)
            except Exception as e:
                exceptions.append(e)

        # Start multiple threads changing state with debug logging
        threads = []
        test_cases = [
            (ApplicationState.RECORDING, "concurrent_test_1"),
            (ApplicationState.ERROR, "concurrent_test_2"),
            (ApplicationState.STOPPING, "concurrent_test_3"),
        ]

        for state, trigger in test_cases:
            thread = threading.Thread(target=change_state_worker, args=(state, trigger))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no exceptions occurred
        assert len(exceptions) == 0, f"Thread safety compromised: {exceptions}"

        # Verify final state is one of the expected states
        expected_states = [case[0] for case in test_cases]
        assert self.state_machine.current_state in expected_states


if __name__ == "__main__":
    pytest.main([__file__, "-v"])