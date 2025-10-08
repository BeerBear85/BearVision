"""
Test script for Edge Application Configuration System

This script tests the configuration loading and parameter system.
"""

import sys
from pathlib import Path

# Add module paths
MODULE_DIR = Path(__file__).resolve().parent.parent.parent / "code" / "modules"
sys.path.append(str(MODULE_DIR))

from EdgeApplicationConfig import EdgeApplicationConfig


def test_default_config():
    """Test default configuration values."""
    print("=" * 70)
    print("Test 1: Default Configuration")
    print("=" * 70)

    config = EdgeApplicationConfig()

    print(f"YOLO Enabled: {config.get_yolo_enabled()}")
    print(f"YOLO Model: {config.get_yolo_model()}")
    print(f"Recording Duration: {config.get_recording_duration()}s")
    print(f"Max Error Restarts: {config.get_max_error_restarts()}")
    print(f"Error Restart Delay: {config.get_error_restart_delay()}s")
    print(f"BLE Logging Enabled: {config.get_enable_ble_logging()}")
    print(f"Hindsight Mode Enabled: {config.get_hindsight_mode_enabled()}")
    print(f"Detection Confidence: {config.get_detection_confidence_threshold()}")
    print(f"Detection Cooldown: {config.get_detection_cooldown()}s")

    assert config.get_yolo_enabled() == True, "Default YOLO should be enabled"
    assert config.get_recording_duration() == 30.0, "Default recording duration should be 30s"
    print("\n[PASS] Default configuration test passed\n")


def test_load_from_file():
    """Test loading configuration from INI file."""
    print("=" * 70)
    print("Test 2: Load Configuration from config.ini")
    print("=" * 70)

    config_path = Path(__file__).resolve().parent.parent.parent / "config.ini"
    config = EdgeApplicationConfig()

    success = config.load_from_file(str(config_path))
    assert success, f"Failed to load config from {config_path}"

    print(f"Configuration loaded from: {config_path}")
    config.print_config()

    print("\n[PASS] Configuration file loading test passed\n")


def test_validation():
    """Test configuration validation."""
    print("=" * 70)
    print("Test 3: Configuration Validation")
    print("=" * 70)

    config = EdgeApplicationConfig()

    # Test valid configuration
    assert config.validate(), "Default config should be valid"
    print("[PASS] Default configuration is valid")

    # Test invalid YOLO model
    config._values["yolo_model"] = "invalid_model"
    assert not config.validate(), "Invalid YOLO model should fail validation"
    print("[PASS] Invalid YOLO model correctly rejected")

    # Reset to valid model
    config._values["yolo_model"] = "yolov8n"

    # Test invalid recording duration
    config._values["recording_duration"] = -1.0
    assert not config.validate(), "Negative recording duration should fail"
    print("[PASS] Invalid recording duration correctly rejected")

    # Reset to valid
    config._values["recording_duration"] = 30.0

    # Test invalid confidence threshold
    config._values["detection_confidence_threshold"] = 1.5
    assert not config.validate(), "Confidence > 1.0 should fail"
    print("[PASS] Invalid confidence threshold correctly rejected")

    print("\n[PASS] All validation tests passed\n")


def test_yolo_disabled():
    """Test YOLO disabled configuration."""
    print("=" * 70)
    print("Test 4: YOLO Disabled Configuration")
    print("=" * 70)

    config = EdgeApplicationConfig()
    config._values["yolo_enabled"] = False

    print(f"YOLO Enabled: {config.get_yolo_enabled()}")
    assert config.get_yolo_enabled() == False, "YOLO should be disabled"

    print("[PASS] YOLO can be disabled via configuration")
    print("  Note: When YOLO is disabled, state machine will stay in LOOKING_FOR_WAKEBOARDER")
    print("\n[PASS] YOLO disabled test passed\n")


def test_custom_parameters():
    """Test custom parameter values."""
    print("=" * 70)
    print("Test 5: Custom Parameter Values")
    print("=" * 70)

    config = EdgeApplicationConfig()

    # Modify parameters
    config._values["recording_duration"] = 45.0
    config._values["max_error_restarts"] = 5
    config._values["yolo_model"] = "yolov8m"
    config._values["detection_cooldown"] = 5.0

    print(f"Custom Recording Duration: {config.get_recording_duration()}s")
    print(f"Custom Max Restarts: {config.get_max_error_restarts()}")
    print(f"Custom YOLO Model: {config.get_yolo_model()}")
    print(f"Custom Detection Cooldown: {config.get_detection_cooldown()}s")

    assert config.get_recording_duration() == 45.0
    assert config.get_max_error_restarts() == 5
    assert config.get_yolo_model() == "yolov8m"
    assert config.get_detection_cooldown() == 5.0

    print("\n[PASS] Custom parameter test passed\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Edge Application Configuration System Test Suite")
    print("=" * 70 + "\n")

    try:
        test_default_config()
        test_load_from_file()
        test_validation()
        test_yolo_disabled()
        test_custom_parameters()

        print("=" * 70)
        print("ALL TESTS PASSED [OK]")
        print("=" * 70)
        print("\nEdge Application Configuration System is working correctly!")
        print("\nKey Features:")
        print("  - Configuration loaded from [EDGE_APPLICATION] section in config.ini")
        print("  - YOLO detection can be enabled/disabled")
        print("  - Recording duration configurable")
        print("  - Error recovery behavior configurable")
        print("  - Thread control via configuration")
        print("  - Validation ensures parameter correctness")
        print("=" * 70)

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
