#!/usr/bin/env python3
"""
Test GUI frame pipeline to debug video display issues.
"""

import sys
import time
from pathlib import Path
from PySide6.QtWidgets import QApplication

# Add module paths
MODULE_DIR = Path(__file__).resolve().parents[2] / "code" / "modules"
APP_DIR = Path(__file__).resolve().parents[2] / "code" / "Application"
TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(MODULE_DIR))
sys.path.append(str(APP_DIR))
sys.path.append(str(TOOLS_DIR))

# Import after setting paths
from edge_gui import EDGEMainWindow

def test_gui_frame_pipeline():
    """Test the GUI frame pipeline with debug output."""
    print("=== GUI Frame Pipeline Test ===")
    print("Starting Qt Application...")

    app = QApplication(sys.argv)

    # Create main window
    print("Creating EDGE GUI...")
    main_window = EDGEMainWindow()
    main_window.show()

    print("GUI should now be starting automatic sequence...")
    print("Check console for debug output from frame pipeline...")
    print("Let it run for 10 seconds to capture debug info...")

    # Run for 10 seconds to capture debug output
    import threading
    import time

    def stop_after_delay():
        time.sleep(10)
        app.quit()

    stop_thread = threading.Thread(target=stop_after_delay, daemon=True)
    stop_thread.start()

    # Run the Qt application
    try:
        sys.exit(app.exec_())
    except SystemExit:
        print("\n=== Test Complete ===")
        print("Check the debug output above to see where the frame pipeline breaks")

if __name__ == "__main__":
    test_gui_frame_pipeline()