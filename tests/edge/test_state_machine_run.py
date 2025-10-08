"""
Test to see what happens when we actually RUN the state machine, not just create it.
This will help identify where the GUI is actually failing.
"""
import sys
import logging
import threading
import time
from pathlib import Path

# Add module paths
MODULE_DIR = Path(__file__).resolve().parent.parent.parent / "code" / "modules"
APP_DIR = Path(__file__).resolve().parent.parent.parent / "code" / "Application"
sys.path.append(str(MODULE_DIR))
sys.path.append(str(APP_DIR))

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_state_machine_in_thread():
    """Test running the state machine in a thread like the GUI does."""
    logger.info("=" * 70)
    logger.info("Testing State Machine Execution (like GUI does)")
    logger.info("=" * 70)

    try:
        from edge_application import EdgeApplicationStateMachine
        from EdgeApplicationConfig import EdgeApplicationConfig
        from EdgeStateMachine import ApplicationState

        config_path = Path(__file__).resolve().parent / "config.ini"
        config = EdgeApplicationConfig()
        if config_path.exists():
            config.load_from_file(str(config_path))

        state_changes = []

        def status_callback(state: ApplicationState, message: str):
            """Track state changes and messages."""
            timestamp = time.time()
            state_changes.append((timestamp, state, message))
            logger.info(f"[{state.value.upper()}] {message}")

        # Create state machine
        logger.info("Creating state machine...")
        state_machine = EdgeApplicationStateMachine(
            status_callback=status_callback,
            config=config
        )
        logger.info("State machine created successfully")

        # Run state machine in a background thread (like the GUI does)
        logger.info("Starting state machine in background thread...")

        stop_event = threading.Event()
        exception_holder = [None]

        def run_state_machine():
            try:
                state_machine.run()
            except Exception as e:
                exception_holder[0] = e
                logger.error(f"State machine thread error: {e}", exc_info=True)

        state_thread = threading.Thread(target=run_state_machine, daemon=True)
        state_thread.start()
        logger.info("State machine thread started")

        # Monitor for 30 seconds to see what happens
        logger.info("Monitoring state machine for 30 seconds...")
        start_time = time.time()
        last_state = None

        while time.time() - start_time < 30:
            time.sleep(1)
            elapsed = time.time() - start_time
            current_state = state_machine.get_state()

            if current_state != last_state:
                logger.info(f"[{elapsed:.1f}s] Current state: {current_state.value}")
                last_state = current_state

            # Check if thread crashed
            if not state_thread.is_alive():
                logger.error("State machine thread has stopped!")
                if exception_holder[0]:
                    logger.error(f"Thread exception: {exception_holder[0]}")
                break

            # Check if we're stuck in ERROR or STOPPING
            if current_state == ApplicationState.ERROR:
                logger.error("State machine is in ERROR state!")
                logger.info(f"Error message: {state_machine.state_machine.error_message}")
                break
            elif current_state == ApplicationState.STOPPING:
                logger.info("State machine is stopping")
                break

        logger.info("\n" + "=" * 70)
        logger.info("State Machine Execution Summary")
        logger.info("=" * 70)
        logger.info(f"Final state: {state_machine.get_state().value}")
        logger.info(f"Total state changes: {len(state_changes)}")
        logger.info("\nState change timeline:")
        for i, (ts, state, msg) in enumerate(state_changes):
            logger.info(f"  {i+1}. [{ts - (state_changes[0][0] if state_changes else ts):.2f}s] {state.value}: {msg}")

        # Shutdown
        logger.info("\nShutting down state machine...")
        state_machine.shutdown()
        state_thread.join(timeout=5)

        if state_thread.is_alive():
            logger.warning("State machine thread did not stop cleanly")
        else:
            logger.info("State machine thread stopped successfully")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

if __name__ == "__main__":
    test_state_machine_in_thread()
