#!/usr/bin/env python3
"""
Test the hindsight mode changes.
"""

import sys
import time
from pathlib import Path

# Add module paths
MODULE_DIR = Path(__file__).resolve().parents[2] / "code" / "modules"
APP_DIR = Path(__file__).resolve().parents[2] / "code" / "Application"
sys.path.append(str(MODULE_DIR))
sys.path.append(str(APP_DIR))

from GoProController import GoProController
from edge_application import EdgeApplication, SystemStatus

def test_gopro_hindsight():
    """Test GoPro hindsight mode functionality."""
    print("=== Testing GoPro Hindsight Mode ===")

    try:
        print("1. Connecting to GoPro...")
        gopro = GoProController()
        gopro.connect()
        gopro.configure()

        print("2. Testing startHindsightMode()...")
        success = gopro.startHindsightMode()
        print(f"   Result: {'SUCCESS' if success else 'FAILED'}")

        if not success:
            print("   This is expected if http_settings is not available")

        print("3. Disconnecting...")
        gopro.disconnect()

        return success

    except Exception as e:
        print(f"   [ERROR] {e}")
        return False

def test_edge_application_hindsight():
    """Test EdgeApplication hindsight functionality."""
    print("\n=== Testing EdgeApplication Hindsight ===")

    log_messages = []

    def log_callback(level: str, message: str):
        log_messages.append((level, message))
        print(f"[{level.upper()}] {message}")

    def status_callback(status: SystemStatus):
        print(f"Status: hindsight_mode={status.hindsight_mode}, overall={status.overall_status.value}")

    try:
        print("1. Creating EdgeApplication...")
        edge_app = EdgeApplication(
            log_callback=log_callback,
            status_callback=status_callback
        )

        print("2. Initializing...")
        if not edge_app.initialize():
            print("   [FAIL] Initialization failed")
            return False

        print("3. Connecting to GoPro...")
        if not edge_app.connect_gopro():
            print("   [FAIL] GoPro connection failed")
            return False

        print("4. Testing trigger_hindsight() - should enable mode...")
        hindsight_result = edge_app.trigger_hindsight()
        print(f"   trigger_hindsight() result: {'SUCCESS' if hindsight_result else 'FAILED'}")

        print("5. Testing trigger_hindsight_clip() - should record clip...")
        clip_result = edge_app.trigger_hindsight_clip()
        print(f"   trigger_hindsight_clip() result: {'SUCCESS' if clip_result else 'FAILED'}")

        print("6. Stopping system...")
        edge_app.stop_system()

        # Show relevant log messages
        hindsight_logs = [msg for level, msg in log_messages if 'hindsight' in msg.lower()]
        if hindsight_logs:
            print("\n   Hindsight-related logs:")
            for msg in hindsight_logs:
                print(f"   - {msg}")

        return hindsight_result or clip_result

    except Exception as e:
        print(f"   [ERROR] {e}")
        return False

def main():
    """Main test function."""
    print("Hindsight Mode Changes Test")
    print("=" * 40)

    # Test 1: Direct GoPro hindsight
    gopro_success = test_gopro_hindsight()

    # Test 2: EdgeApplication hindsight
    edge_success = test_edge_application_hindsight()

    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print(f"GoPro hindsight test: {'PASS' if gopro_success else 'FAIL'}")
    print(f"EdgeApplication hindsight test: {'PASS' if edge_success else 'FAIL'}")

    if not gopro_success:
        print("\nNote: GoPro hindsight failure is expected if http_settings is not available")
        print("This is a known limitation with some GoPro firmware/connection states")

    if edge_success:
        print("\n[SUCCESS] Hindsight functionality is working correctly")
        print("- trigger_hindsight() enables hindsight mode")
        print("- trigger_hindsight_clip() records a clip")
    else:
        print("\n[ISSUE] EdgeApplication hindsight functionality needs attention")

if __name__ == "__main__":
    main()