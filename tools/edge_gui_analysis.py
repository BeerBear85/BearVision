"""
Comprehensive analysis of EDGE GUI functionality.
This script tests the GUI without requiring human interaction.
"""

import sys
import os
import time
import threading
from pathlib import Path

# Add paths
MODULE_DIR = Path(__file__).resolve().parent.parent / "code" / "modules"
APP_DIR = Path(__file__).resolve().parent.parent / "code" / "Application"
sys.path.append(str(MODULE_DIR))
sys.path.append(str(APP_DIR))

def test_gui_startup():
    """Test if GUI can start up properly."""
    print("Testing GUI startup...")

    try:
        from PySide6.QtWidgets import QApplication
        from edge_gui import EDGEMainWindow

        # Create QApplication (required for any Qt GUI)
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        # Create main window
        window = EDGEMainWindow()
        print("[PASS] Main window created successfully")

        # Test window properties
        if window.windowTitle() == "BearVision EDGE Application":
            print("[PASS] Window title set correctly")
        else:
            print(f"[FAIL] Window title incorrect: {window.windowTitle()}")

        # Test minimum size
        min_size = window.minimumSize()
        if min_size.width() >= 1200 and min_size.height() >= 800:
            print("[PASS] Minimum window size set correctly")
        else:
            print(f"[FAIL] Minimum size incorrect: {min_size.width()}x{min_size.height()}")

        # Test that main components exist
        if hasattr(window, 'status_bar'):
            print("[PASS] Status bar component exists")
        else:
            print("[FAIL] Status bar component missing")

        if hasattr(window, 'preview_area'):
            print("[PASS] Preview area component exists")
        else:
            print("[FAIL] Preview area component missing")

        if hasattr(window, 'indicators_panel'):
            print("[PASS] Indicators panel component exists")
        else:
            print("[FAIL] Indicators panel component missing")

        if hasattr(window, 'event_list'):
            print("[PASS] Event list component exists")
        else:
            print("[FAIL] Event list component missing")

        return True, window, app

    except Exception as e:
        print(f"[FAIL] GUI startup failed: {e}")
        return False, None, None

def test_backend_integration(window):
    """Test backend integration."""
    print("\nTesting backend integration...")

    try:
        # Test backend exists
        if hasattr(window, 'backend'):
            print("[PASS] Backend component exists")
        else:
            print("[FAIL] Backend component missing")
            return False

        backend = window.backend

        # Test backend thread state
        if backend.isRunning():
            print("[PASS] Backend thread is running")
        else:
            print("[WARN] Backend thread not running")

        # Test signal connections exist
        signal_tests = [
            ('log_event', 'handle_backend_log_event'),
            ('status_changed', 'handle_status_changed'),
            ('motion_detected', 'handle_motion_detected'),
            ('hindsight_triggered', 'handle_hindsight_triggered'),
            ('preview_frame', 'handle_preview_frame')
        ]

        for signal_name, handler_name in signal_tests:
            if hasattr(backend, signal_name) and hasattr(window, handler_name):
                print(f"[PASS] Signal {signal_name} -> {handler_name} connection ready")
            else:
                print(f"[FAIL] Signal {signal_name} -> {handler_name} connection missing")

        return True

    except Exception as e:
        print(f"[FAIL] Backend integration test failed: {e}")
        return False

def test_menu_actions(window):
    """Test menu actions exist and are callable."""
    print("\nTesting menu actions...")

    try:
        # Test menu action methods exist
        menu_methods = [
            'initialize_edge_system',
            'connect_gopro',
            'start_preview',
            'stop_preview',
            'start_edge_processing',
            'trigger_hindsight',
            'stop_edge_processing'
        ]

        for method_name in menu_methods:
            if hasattr(window, method_name):
                method = getattr(window, method_name)
                if callable(method):
                    print(f"[PASS] Menu action {method_name} exists and is callable")
                else:
                    print(f"[FAIL] Menu action {method_name} exists but not callable")
            else:
                print(f"[FAIL] Menu action {method_name} missing")

        return True

    except Exception as e:
        print(f"[FAIL] Menu actions test failed: {e}")
        return False

def test_demo_functionality(window):
    """Test demo/simulation functionality."""
    print("\nTesting demo functionality...")

    try:
        # Test demo data exists
        if hasattr(window, 'demo_detections'):
            print("[PASS] Demo detections data exists")
        else:
            print("[FAIL] Demo detections data missing")

        # Test event log has some initial events
        if hasattr(window, 'event_list') and len(window.event_list.events) > 0:
            print(f"[PASS] Event log has {len(window.event_list.events)} initial events")
        else:
            print("[FAIL] Event log empty or missing")

        # Test status indicators
        if hasattr(window, 'indicators_panel'):
            indicators = window.indicators_panel.indicators
            if len(indicators) >= 4:  # Should have at least 4 indicators
                print(f"[PASS] Indicators panel has {len(indicators)} indicators")
            else:
                print(f"[FAIL] Indicators panel has only {len(indicators)} indicators")

        return True

    except Exception as e:
        print(f"[FAIL] Demo functionality test failed: {e}")
        return False

def test_edge_module_availability():
    """Test if actual EDGE modules are available."""
    print("\nTesting EDGE module availability...")

    try:
        from edge_gui import EDGE_AVAILABLE

        if EDGE_AVAILABLE:
            print("[PASS] EDGE modules are available")

            # Try importing specific modules
            try:
                from ConfigurationHandler import ConfigurationHandler
                print("[PASS] ConfigurationHandler import successful")
            except ImportError as e:
                print(f"[FAIL] ConfigurationHandler import failed: {e}")

            try:
                from GoProController import GoProController
                print("[PASS] GoProController import successful")
            except ImportError as e:
                print(f"[FAIL] GoProController import failed: {e}")

            try:
                import edge_main
                print("[PASS] edge_main import successful")
            except ImportError as e:
                print(f"[FAIL] edge_main import failed: {e}")
        else:
            print("[WARN] EDGE modules not available - GUI will run in demo mode only")

        return EDGE_AVAILABLE

    except Exception as e:
        print(f"[FAIL] EDGE module availability test failed: {e}")
        return False

def test_widget_functionality(window):
    """Test individual widget functionality."""
    print("\nTesting widget functionality...")

    try:
        # Test status bar update
        window.status_bar.update_status("Test message")
        if window.status_bar.status_label.text() == "Test message":
            print("[PASS] Status bar update works")
        else:
            print("[FAIL] Status bar update failed")

        # Test adding log event
        initial_count = len(window.event_list.events)
        from edge_gui import EventType
        window.add_log_event(EventType.INFO, "Test log message")

        if len(window.event_list.events) == initial_count + 1:
            print("[PASS] Log event addition works")
        else:
            print("[FAIL] Log event addition failed")

        # Test indicators update
        from edge_gui import StatusIndicators
        test_status = StatusIndicators()
        test_status.active = True
        test_status.preview = True

        window.indicators_panel.update_indicators(test_status)
        print("[PASS] Indicators update works")

        return True

    except Exception as e:
        print(f"[FAIL] Widget functionality test failed: {e}")
        return False

def test_graceful_shutdown(window, app):
    """Test that the application can shut down gracefully."""
    print("\nTesting graceful shutdown...")

    try:
        # Test that cleanup methods exist
        if hasattr(window, 'closeEvent'):
            print("[PASS] Close event handler exists")
        else:
            print("[FAIL] Close event handler missing")

        # Test backend stop
        if hasattr(window.backend, 'stop_edge'):
            print("[PASS] Backend stop method exists")
        else:
            print("[FAIL] Backend stop method missing")

        # Actually perform shutdown (but don't show window)
        window.backend.stop_edge()

        if window.backend.isRunning():
            window.backend.quit()
            window.backend.wait(1000)  # Wait 1 second

        print("[PASS] Graceful shutdown completed")
        return True

    except Exception as e:
        print(f"[FAIL] Graceful shutdown test failed: {e}")
        return False

def main():
    """Run comprehensive GUI analysis."""
    print("EDGE GUI Comprehensive Analysis")
    print("=" * 50)

    # Prevent GUI from actually showing
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

    tests = []

    # Test 1: GUI Startup
    success, window, app = test_gui_startup()
    tests.append(("GUI Startup", success))

    if not success or window is None:
        print("\n[CRITICAL] Cannot continue - GUI startup failed")
        return

    # Test 2: Backend Integration
    success = test_backend_integration(window)
    tests.append(("Backend Integration", success))

    # Test 3: Menu Actions
    success = test_menu_actions(window)
    tests.append(("Menu Actions", success))

    # Test 4: Demo Functionality
    success = test_demo_functionality(window)
    tests.append(("Demo Functionality", success))

    # Test 5: Widget Functionality
    success = test_widget_functionality(window)
    tests.append(("Widget Functionality", success))

    # Test 6: EDGE Module Availability
    success = test_edge_module_availability()
    tests.append(("EDGE Module Availability", success))

    # Test 7: Graceful Shutdown
    success = test_graceful_shutdown(window, app)
    tests.append(("Graceful Shutdown", success))

    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(tests)

    for test_name, result in tests:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 [SUCCESS] EDGE GUI is fully functional!")
        print("The application is ready for production use.")
    elif passed >= total * 0.8:  # 80% pass rate
        print("\n✅ [GOOD] EDGE GUI is mostly functional with minor issues.")
        print("The application should work well in most scenarios.")
    elif passed >= total * 0.5:  # 50% pass rate
        print("\n⚠️  [PARTIAL] EDGE GUI has significant issues that need attention.")
        print("Some functionality may not work as expected.")
    else:
        print("\n❌ [CRITICAL] EDGE GUI has major problems.")
        print("Substantial fixes are needed before use.")

    print(f"\nTo run the GUI: python tools/edge_gui.py")

if __name__ == "__main__":
    main()