"""
Test script for EDGE GUI - validates the implementation without opening the GUI
"""

import sys
from pathlib import Path

# Add paths
MODULE_DIR = Path(__file__).resolve().parent.parent / "code" / "modules"
APP_DIR = Path(__file__).resolve().parent.parent / "code" / "Application"
sys.path.append(str(MODULE_DIR))
sys.path.append(str(APP_DIR))

def test_edge_gui_imports():
    """Test that all components can be imported."""
    try:
        # Test imports
        from edge_gui import (
            EventType, DetectionBox, Event, StatusIndicators,
            EDGEBackend, StatusBar, PreviewArea, IndicatorWidget,
            IndicatorsPanel, EventList, EDGEMainWindow
        )
        print("[PASS] All GUI components imported successfully")
        return True
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False

def test_backend_creation():
    """Test backend creation without starting."""
    try:
        from edge_gui import EDGEBackend, EventType, StatusIndicators

        backend = EDGEBackend()
        print("[PASS] EDGEBackend created successfully")

        # Test status indicators
        status = StatusIndicators()
        status.active = True
        print("[PASS] StatusIndicators created successfully")

        return True
    except Exception as e:
        print(f"[FAIL] Backend creation error: {e}")
        return False

def test_data_structures():
    """Test data structure creation."""
    try:
        from edge_gui import EventType, DetectionBox, Event
        from datetime import datetime

        # Test event creation
        event = Event("1", "12:34:56", EventType.SUCCESS, "Test message")
        print("[PASS] Event creation successful")

        # Test detection box
        detection = DetectionBox("test", 10.0, 20.0, 5.0, 8.0, "Person", 0.85)
        print("[PASS] DetectionBox creation successful")

        return True
    except Exception as e:
        print(f"[FAIL] Data structure error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing EDGE GUI Implementation")
    print("=" * 40)

    tests = [
        test_edge_gui_imports,
        test_backend_creation,
        test_data_structures
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[FAIL] Test {test.__name__} failed: {e}")

    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("[SUCCESS] All tests passed! EDGE GUI is ready to use.")
        print("\nTo run the GUI application:")
        print("python tools/edge_gui.py")
    else:
        print("[ERROR] Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()