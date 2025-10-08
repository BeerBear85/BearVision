"""
Test specifically to see why hindsight mode fails.
"""
import sys
import logging
from pathlib import Path

# Add module paths
MODULE_DIR = Path(__file__).resolve().parent / "code" / "modules"
sys.path.append(str(MODULE_DIR))

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_hindsight_mode():
    """Test hindsight mode enabling."""
    logger.info("=" * 70)
    logger.info("Testing Hindsight Mode Enablement")
    logger.info("=" * 70)

    try:
        from GoProController import GoProController

        logger.info("Connecting to GoPro...")
        gopro = GoProController()
        gopro.connect()
        logger.info("GoPro connected successfully")

        logger.info("\nConfiguring GoPro...")
        gopro.configure()
        logger.info("GoPro configuration completed")

        # Check if http_settings is available
        logger.info(f"\nChecking for http_settings attribute: {hasattr(gopro._gopro, 'http_settings')}")

        if hasattr(gopro._gopro, 'http_settings'):
            logger.info("http_settings IS available")
            logger.info(f"http_settings type: {type(gopro._gopro.http_settings)}")
            logger.info(f"hindsight attribute: {hasattr(gopro._gopro.http_settings, 'hindsight')}")

            if hasattr(gopro._gopro.http_settings, 'hindsight'):
                logger.info(f"hindsight type: {type(gopro._gopro.http_settings.hindsight)}")
        else:
            logger.warning("http_settings is NOT available!")

            # Try to see what attributes ARE available
            logger.info(f"\nAvailable _gopro attributes: {dir(gopro._gopro)}")

        logger.info("\nAttempting to enable hindsight mode...")
        result = gopro.startHindsightMode()
        logger.info(f"Hindsight mode enable result: {result}")

        if not result:
            logger.error("Hindsight mode failed to enable!")

            # Check what the actual issue is
            logger.info("\nInvestigating why hindsight failed...")
            if not hasattr(gopro._gopro, 'http_settings'):
                logger.error("ROOT CAUSE: http_settings attribute not available on GoPro object")
            else:
                logger.info("http_settings exists, but something else went wrong")
        else:
            logger.info("Hindsight mode enabled successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

if __name__ == "__main__":
    test_hindsight_mode()
