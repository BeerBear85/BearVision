"""
Quick verification that the fix resolves the edge GUI startup issue.
"""
import sys
import logging
import time
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent.parent.parent / "code" / "modules"
APP_DIR = Path(__file__).resolve().parent.parent.parent / "code" / "Application"
sys.path.append(str(MODULE_DIR))
sys.path.append(str(APP_DIR))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_fix():
    """Test that state machine reaches LOOKING_FOR_WAKEBOARDER state."""
    from edge_application import EdgeApplicationStateMachine
    from EdgeApplicationConfig import EdgeApplicationConfig
    from EdgeStateMachine import ApplicationState
    import threading

    logger.info("=" * 70)
    logger.info("VERIFICATION TEST: Edge GUI Startup Fix")
    logger.info("=" * 70)

    config_path = Path(__file__).resolve().parent.parent.parent / "config.ini"
    config = EdgeApplicationConfig()
    config.load_from_file(str(config_path))

    states_reached = []

    def status_callback(state: ApplicationState, message: str):
        states_reached.append((state, message))
        if state == ApplicationState.LOOKING_FOR_WAKEBOARDER and "Hindsight mode enabled" in message:
            logger.info(f"✓ SUCCESS: Reached LOOKING_FOR_WAKEBOARDER with hindsight enabled!")
        elif state == ApplicationState.ERROR:
            logger.error(f"✗ FAILURE: Entered ERROR state: {message}")

    state_machine = EdgeApplicationStateMachine(
        status_callback=status_callback,
        config=config
    )

    def run_sm():
        try:
            state_machine.run()
        except:
            pass

    thread = threading.Thread(target=run_sm, daemon=True)
    thread.start()

    # Wait up to 15 seconds for initialization
    logger.info("\nWaiting for initialization...")
    for i in range(15):
        time.sleep(1)
        current_state = state_machine.get_state()

        if current_state == ApplicationState.LOOKING_FOR_WAKEBOARDER:
            logger.info(f"✓ State machine reached LOOKING_FOR_WAKEBOARDER in {i+1} seconds")
            logger.info("✓ FIX VERIFIED: Edge GUI startup issue is RESOLVED!")
            state_machine.shutdown()
            return True
        elif current_state == ApplicationState.ERROR:
            logger.error(f"✗ State machine entered ERROR state")
            logger.error(f"Error: {state_machine.state_machine.error_message}")
            logger.error("✗ FIX FAILED: Issue still exists")
            state_machine.shutdown()
            return False

    logger.error("✗ Timeout waiting for LOOKING_FOR_WAKEBOARDER state")
    state_machine.shutdown()
    return False

if __name__ == "__main__":
    success = test_fix()
    sys.exit(0 if success else 1)
