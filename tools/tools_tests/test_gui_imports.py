"""Test GUI imports and basic initialization"""
import sys
from pathlib import Path

# Add paths
MODULE_DIR = Path(__file__).resolve().parent.parent.parent / "code" / "modules"
APP_DIR = Path(__file__).resolve().parent.parent.parent / "code" / "Application"
sys.path.append(str(MODULE_DIR))
sys.path.append(str(APP_DIR))

def test_gui_imports():
    """Test that GUI imports work correctly."""
    print("Testing imports...")

    from edge_application import EdgeApplicationStateMachine
    print("[OK] EdgeApplicationStateMachine imported")

    from EdgeStateMachine import ApplicationState
    print("[OK] ApplicationState imported")

    from StatusManager import SystemStatus, EdgeStatus, DetectionResult
    print("[OK] StatusManager imports successful")

    from EdgeApplicationConfig import EdgeApplicationConfig
    print("[OK] EdgeApplicationConfig imported")

    # Test creating a config
    config = EdgeApplicationConfig()
    print("[OK] EdgeApplicationConfig created")

    # Test creating state machine
    def status_callback(state, message):
        print(f"  State: {state}, Message: {message}")

    sm = EdgeApplicationStateMachine(status_callback=status_callback, config=config)
    print("[OK] EdgeApplicationStateMachine created")

    print("\nAll imports and basic initialization successful!")
    print("State machine architecture is correctly integrated.")

if __name__ == "__main__":
    test_gui_imports()
