"""
Visual test for EDGE GUI - actually shows the GUI window briefly for verification
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
    """Show GUI briefly for visual verification."""
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QTimer
        from edge_gui import EDGEMainWindow

        print("Starting EDGE GUI visual test...")
        print("The GUI window will appear for 5 seconds, then close automatically.")
        print("This verifies that the GUI renders correctly.")

        # Create application
        app = QApplication(sys.argv)

        # Create main window
        window = EDGEMainWindow()
        window.show()

        print("GUI window is now visible...")
        print("Window components should include:")
        print("- Status bar with BearVision logo and status message")
        print("- Video preview area (gray with demo image)")
        print("- Status indicators panel on the right")
        print("- Event log panel at the bottom right")
        print("- Menu bar with File and Tools menus")

        # Auto-close after 5 seconds
        def close_window():
            print("Closing window...")
            window.close()
            app.quit()

        timer = QTimer()
        timer.timeout.connect(close_window)
        timer.start(5000)  # 5 seconds

        # Run the application
        result = app.exec()
        print("GUI test completed successfully!")
        return result

    except Exception as e:
        print(f"Visual test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())