"""
Verification test for edge_gui menu restructuring.
Tests that all menu items are properly configured and state transitions work.
"""

import sys
import io
from pathlib import Path

# Set UTF-8 encoding for stdout on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add module paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code" / "modules"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code" / "Application"))

from PySide6.QtWidgets import QApplication
from edge_gui import EDGEMainWindow, EDGEBackend
from EdgeStateMachine import ApplicationState

def test_menu_structure():
    """Test that menu structure is correct."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = EDGEMainWindow()

    # Check menu bar exists
    menubar = window.menuBar()
    menu_names = [action.text() for action in menubar.actions()]

    print("[OK] Menu bar created")
    print(f"  Menus: {menu_names}")

    # Verify expected menus
    expected_menus = ["File", "Camera", "System", "Demo"]
    for menu in expected_menus:
        assert menu in menu_names, f"Missing menu: {menu}"

    print("[OK] All expected menus present")

    # Check Camera menu items
    camera_menu = None
    for action in menubar.actions():
        if action.text() == "Camera":
            camera_menu = action.menu()
            break

    assert camera_menu is not None, "Camera menu not found"
    camera_actions = [action.text() for action in camera_menu.actions() if action.text()]
    print(f"  Camera menu items: {camera_actions}")

    assert "🖥️ Preview" in camera_actions, "Preview action not found"
    assert "📹 Capture Clip Now" in camera_actions, "Capture action not found"

    print("[OK] Camera menu items correct")

    # Check System menu items
    system_menu = None
    for action in menubar.actions():
        if action.text() == "System":
            system_menu = action.menu()
            break

    assert system_menu is not None, "System menu not found"
    system_actions = [action.text() for action in system_menu.actions() if action.text()]
    print(f"  System menu items: {system_actions}")

    assert "▶️ Start Auto-Capture" in system_actions, "Auto-capture action not found"
    assert "🔄 Restart System" in system_actions, "Restart action not found"

    print("[OK] System menu items correct")

    # Check that menu action attributes exist
    assert hasattr(window, 'preview_action'), "preview_action not found"
    assert hasattr(window, 'capture_action'), "capture_action not found"
    assert hasattr(window, 'auto_capture_action'), "auto_capture_action not found"
    assert hasattr(window, 'restart_action'), "restart_action not found"

    print("[OK] Menu action attributes exist")

    # Check menu action methods exist
    assert hasattr(window, 'toggle_preview'), "toggle_preview method not found"
    assert hasattr(window, 'capture_clip_now'), "capture_clip_now method not found"
    assert hasattr(window, 'toggle_auto_capture'), "toggle_auto_capture method not found"
    assert hasattr(window, 'start_auto_capture'), "start_auto_capture method not found"
    assert hasattr(window, 'stop_auto_capture'), "stop_auto_capture method not found"
    assert hasattr(window, 'restart_system'), "restart_system method not found"

    print("[OK] Menu action methods exist")

    # Check state handling methods exist
    assert hasattr(window, 'handle_state_changed'), "handle_state_changed method not found"
    assert hasattr(window, 'update_menu_for_state'), "update_menu_for_state method not found"

    print("[OK] State handling methods exist")

    # Check backend signal exists
    assert hasattr(window.backend, 'state_changed'), "state_changed signal not found"

    print("[OK] Backend state_changed signal exists")

    # Check state tracking variables
    assert hasattr(window, 'current_state'), "current_state variable not found"
    assert hasattr(window, 'system_running'), "system_running variable not found"

    print("[OK] State tracking variables exist")

    # Test initial menu state (should be disabled until system starts)
    assert not window.preview_action.isEnabled(), "Preview should be disabled initially"
    assert not window.capture_action.isEnabled(), "Capture should be disabled initially"
    assert not window.restart_action.isEnabled(), "Restart should be disabled initially"

    print("[OK] Initial menu state correct (all disabled)")

    # Check shortcuts
    assert window.preview_action.shortcut().toString() == "Ctrl+P", "Preview shortcut incorrect"
    assert window.capture_action.shortcut().toString() == "Ctrl+R", "Capture shortcut incorrect"
    assert window.auto_capture_action.shortcut().toString() == "Ctrl+Q", "Auto-capture shortcut incorrect"

    print("[OK] Keyboard shortcuts configured")

    # Check tooltips
    assert window.preview_action.toolTip(), "Preview tooltip missing"
    assert window.capture_action.toolTip(), "Capture tooltip missing"
    assert window.auto_capture_action.toolTip(), "Auto-capture tooltip missing"
    assert window.restart_action.toolTip(), "Restart tooltip missing"

    print("[OK] Tooltips configured")

    # Test state transition logic (without actual backend)
    window.update_menu_for_state(ApplicationState.LOOKING_FOR_WAKEBOARDER)
    assert window.preview_action.isEnabled(), "Preview should be enabled in LOOKING_FOR_WAKEBOARDER"
    assert window.capture_action.isEnabled(), "Capture should be enabled in LOOKING_FOR_WAKEBOARDER"
    assert not window.restart_action.isEnabled(), "Restart should be disabled in LOOKING_FOR_WAKEBOARDER"

    print("[OK] LOOKING_FOR_WAKEBOARDER state menu updates correct")

    window.update_menu_for_state(ApplicationState.RECORDING)
    assert window.preview_action.isEnabled(), "Preview should be enabled in RECORDING"
    assert not window.capture_action.isEnabled(), "Capture should be disabled in RECORDING"
    assert not window.restart_action.isEnabled(), "Restart should be disabled in RECORDING"

    print("[OK] RECORDING state menu updates correct")

    window.update_menu_for_state(ApplicationState.ERROR)
    assert not window.preview_action.isEnabled(), "Preview should be disabled in ERROR"
    assert not window.capture_action.isEnabled(), "Capture should be disabled in ERROR"
    assert window.restart_action.isEnabled(), "Restart should be enabled in ERROR"

    print("[OK] ERROR state menu updates correct")

    print("\n" + "="*60)
    print("[SUCCESS] ALL TESTS PASSED - Menu restructuring verified!")
    print("="*60)

    return True

if __name__ == "__main__":
    try:
        test_menu_structure()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
