"""
Quick test script to diagnose edge GUI startup failure.
"""
import sys
import logging
from pathlib import Path

# Add module paths
MODULE_DIR = Path(__file__).resolve().parent.parent.parent / "code" / "modules"
APP_DIR = Path(__file__).resolve().parent.parent.parent / "code" / "Application"
sys.path.append(str(MODULE_DIR))
sys.path.append(str(APP_DIR))

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported."""
    logger.info("Testing imports...")
    try:
        from ConfigurationHandler import ConfigurationHandler
        logger.info("✓ ConfigurationHandler imported")
    except Exception as e:
        logger.error(f"✗ ConfigurationHandler: {e}")

    try:
        from GoProController import GoProController
        logger.info("✓ GoProController imported")
    except Exception as e:
        logger.error(f"✗ GoProController: {e}")

    try:
        import edge_main
        logger.info("✓ edge_main imported")
    except Exception as e:
        logger.error(f"✗ edge_main: {e}")

    try:
        from edge_application import EdgeApplicationStateMachine
        logger.info("✓ EdgeApplicationStateMachine imported")
    except Exception as e:
        logger.error(f"✗ EdgeApplicationStateMachine: {e}")

    try:
        from EdgeStateMachine import ApplicationState
        logger.info("✓ ApplicationState imported")
    except Exception as e:
        logger.error(f"✗ ApplicationState: {e}")

    try:
        from StatusManager import SystemStatus, EdgeStatus, DetectionResult
        logger.info("✓ StatusManager imported")
    except Exception as e:
        logger.error(f"✗ StatusManager: {e}")

    try:
        from EdgeApplicationConfig import EdgeApplicationConfig
        logger.info("✓ EdgeApplicationConfig imported")
    except Exception as e:
        logger.error(f"✗ EdgeApplicationConfig: {e}")

def test_config_load():
    """Test loading configuration."""
    logger.info("\nTesting configuration loading...")
    try:
        from EdgeApplicationConfig import EdgeApplicationConfig
        config_path = Path(__file__).resolve().parent / "config.ini"
        config = EdgeApplicationConfig()
        if config.load_from_file(str(config_path)):
            logger.info(f"✓ Configuration loaded from {config_path}")
            config.print_config()
        else:
            logger.warning("⚠ Using default configuration values")
    except Exception as e:
        logger.error(f"✗ Configuration loading failed: {e}")
        import traceback
        traceback.print_exc()

def test_edge_app_creation():
    """Test creating EdgeApplicationStateMachine."""
    logger.info("\nTesting EdgeApplicationStateMachine creation...")
    try:
        from edge_application import EdgeApplicationStateMachine
        from EdgeApplicationConfig import EdgeApplicationConfig
        from EdgeStateMachine import ApplicationState

        config_path = Path(__file__).resolve().parent / "config.ini"
        config = EdgeApplicationConfig()
        if config_path.exists():
            config.load_from_file(str(config_path))

        def status_callback(state: ApplicationState, message: str):
            logger.info(f"[{state.value.upper()}] {message}")

        state_machine = EdgeApplicationStateMachine(
            status_callback=status_callback,
            config=config
        )
        logger.info("✓ EdgeApplicationStateMachine created successfully")

        # Check if edge_app exists
        if state_machine.edge_app:
            logger.info("✓ edge_app initialized")
        else:
            logger.error("✗ edge_app is None")

    except Exception as e:
        logger.error(f"✗ EdgeApplicationStateMachine creation failed: {e}")
        import traceback
        traceback.print_exc()

def test_gopro_detection():
    """Test if GoPro can be detected."""
    logger.info("\nTesting GoPro detection...")
    try:
        from GoProController import GoProController
        gopro = GoProController()
        logger.info("Attempting to connect to GoPro...")
        gopro.connect()
        logger.info("✓ GoPro connection initiated")
    except Exception as e:
        logger.error(f"✗ GoPro connection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Edge GUI Startup Diagnostic")
    logger.info("=" * 70)

    test_imports()
    test_config_load()
    test_edge_app_creation()
    test_gopro_detection()

    logger.info("=" * 70)
    logger.info("Diagnostic complete")
    logger.info("=" * 70)
