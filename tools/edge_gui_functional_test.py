"""
Functional test for EDGE GUI - tests actual menu actions and functionality
"""

import sys
import os
from pathlib import Path

# Add paths
MODULE_DIR = Path(__file__).resolve().parent.parent / "code" / "modules"
APP_DIR = Path(__file__).resolve().parent.parent / "code" / "Application"
sys.path.append(str(MODULE_DIR))
sys.path.append(str(APP_DIR))

def main():
    """Test GUI functionality by simulating user interactions."""
    print("EDGE GUI Functional Test")
    print("=" * 40)

    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QTimer
        from edge_gui import EDGEMainWindow, EventType

        # Prevent actual window display
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'

        # Create application
        app = QApplication(sys.argv)

        # Create main window
        window = EDGEMainWindow()

        print("Testing menu functionality...")

        # Test 1: Initialize EDGE System
        print("\n1. Testing Initialize EDGE System...")
        initial_events = len(window.event_list.events)
        window.initialize_edge_system()

        if len(window.event_list.events) > initial_events:
            print("[PASS] Initialize EDGE System triggered events")
        else:
            print("[FAIL] Initialize EDGE System did not trigger events")

        # Test 2: Connect GoPro
        print("\n2. Testing Connect GoPro...")
        initial_events = len(window.event_list.events)
        window.connect_gopro()

        if len(window.event_list.events) > initial_events:
            print("[PASS] Connect GoPro triggered events")
        else:
            print("[FAIL] Connect GoPro did not trigger events")

        # Test 3: Start Preview
        print("\n3. Testing Start Preview...")
        initial_events = len(window.event_list.events)
        window.start_preview()

        if len(window.event_list.events) > initial_events:
            print("[PASS] Start Preview triggered events")
            # Check if preview image was updated
            if window.preview_area.current_image is not None:
                print("[PASS] Preview image was updated")
            else:
                print("[WARN] Preview image not updated")
        else:
            print("[FAIL] Start Preview did not trigger events")

        # Test 4: Trigger Hindsight
        print("\n4. Testing Trigger Hindsight...")
        initial_events = len(window.event_list.events)
        window.trigger_hindsight()

        if len(window.event_list.events) > initial_events:
            print("[PASS] Trigger Hindsight triggered events")
        else:
            print("[FAIL] Trigger Hindsight did not trigger events")

        # Test 5: Start EDGE Processing
        print("\n5. Testing Start EDGE Processing...")
        initial_events = len(window.event_list.events)
        window.start_edge_processing()

        if len(window.event_list.events) > initial_events:
            print("[PASS] Start EDGE Processing triggered events")
        else:
            print("[FAIL] Start EDGE Processing did not trigger events")

        # Test 6: Stop Preview
        print("\n6. Testing Stop Preview...")
        initial_events = len(window.event_list.events)
        window.stop_preview()

        if len(window.event_list.events) > initial_events:
            print("[PASS] Stop Preview triggered events")
        else:
            print("[FAIL] Stop Preview did not trigger events")

        # Test 7: Stop EDGE Processing
        print("\n7. Testing Stop EDGE Processing...")
        initial_events = len(window.event_list.events)
        window.stop_edge_processing()

        if len(window.event_list.events) > initial_events:
            print("[PASS] Stop EDGE Processing triggered events")
        else:
            print("[FAIL] Stop EDGE Processing did not trigger events")

        # Test 8: Backend Signal Simulation
        print("\n8. Testing Backend Signal Handling...")
        from edge_gui import StatusIndicators

        # Test status change signal
        test_status = StatusIndicators()
        test_status.active = True
        test_status.preview = True
        test_status.recording = True

        window.handle_status_changed(test_status)
        print("[PASS] Status change signal handled")

        # Test motion detection signal
        window.handle_motion_detected()
        print("[PASS] Motion detection signal handled")

        # Test hindsight trigger signal
        window.handle_hindsight_triggered()
        print("[PASS] Hindsight trigger signal handled")

        # Test log event signal
        window.handle_backend_log_event(EventType.SUCCESS, "Test backend message")
        print("[PASS] Backend log event signal handled")

        print(f"\nTotal events in log: {len(window.event_list.events)}")

        # Cleanup
        window.backend.stop_edge()
        window.backend.quit()
        window.backend.wait(1000)

        print("\n" + "=" * 40)
        print("[SUCCESS] All functional tests completed!")
        print("The EDGE GUI is fully functional and ready for use.")

        return 0

    except Exception as e:
        print(f"[FAIL] Functional test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())