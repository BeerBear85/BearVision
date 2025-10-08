"""
Test edge_main.py with state machine architecture.

This test validates that edge_main correctly uses the EdgeApplicationStateMachine
as specified in the state machine design document.
"""

import sys
from pathlib import Path
from unittest import mock

# Add module paths
MODULE_DIR = Path(__file__).resolve().parents[2] / 'code' / 'modules'
APP_DIR = Path(__file__).resolve().parents[2] / 'code' / 'Application'
sys.path.append(str(MODULE_DIR))
sys.path.append(str(APP_DIR))

import edge_main
from EdgeStateMachine import ApplicationState
from EdgeApplicationConfig import EdgeApplicationConfig


def test_edge_main_uses_state_machine():
    """Test that edge_main uses EdgeApplicationStateMachine."""
    # Test that the main function creates and uses the state machine correctly
    with mock.patch('edge_main.EdgeApplicationStateMachine') as MockStateMachine:
        mock_sm_instance = MockStateMachine.return_value

        # Mock the run method to prevent blocking
        mock_sm_instance.run.side_effect = KeyboardInterrupt()

        # Run main (will exit on KeyboardInterrupt)
        try:
            edge_main.main()
        except SystemExit:
            pass

        # Verify EdgeApplicationStateMachine was created with proper callbacks
        assert MockStateMachine.called
        call_args = MockStateMachine.call_args

        # Should have status_callback and config
        assert 'status_callback' in call_args.kwargs or len(call_args.args) > 0
        assert 'config' in call_args.kwargs or len(call_args.args) > 1

        # Verify run was called
        mock_sm_instance.run.assert_called_once()
        mock_sm_instance.shutdown.assert_called_once()


def test_edge_application_config_loading():
    """Test that configuration is loaded correctly."""
    config = EdgeApplicationConfig()

    # Test default values exist
    assert hasattr(config, 'get_recording_duration')
    assert hasattr(config, 'get_detection_cooldown')
    assert hasattr(config, 'get_hindsight_mode_enabled')
    assert hasattr(config, 'get_yolo_enabled')

    # Test config can be loaded from file (if it exists)
    config_path = Path(__file__).resolve().parents[2] / 'config.ini'
    if config_path.exists():
        result = config.load_from_file(str(config_path))
        # Should return True if file exists and is valid
        assert isinstance(result, bool)
