import logging
import threading
import time
from pathlib import Path
import sys

MODULE_DIR = Path(__file__).resolve().parents[1] / "modules"
sys.path.append(str(MODULE_DIR))

from ConfigurationHandler import ConfigurationHandler
from GoProController import GoProController


logger = logging.getLogger(__name__)


def _preview_algorithm():
    """Placeholder preview algorithm running in a thread."""
    while True:
        time.sleep(0.1)


def _moment_detector():
    """Placeholder moment detection running in a thread."""
    while True:
        time.sleep(0.1)


def main() -> list[threading.Thread]:
    """Load configuration, setup GoPro and start edge threads."""
    logging.basicConfig(level=logging.INFO)

    cfg_path = Path(__file__).resolve().parents[2] / "config.ini"
    ConfigurationHandler.read_config_file(str(cfg_path))
    logger.info("Loaded configuration from %s", cfg_path)

    gopro = GoProController()
    gopro.connect()
    gopro.configure()
    gopro.start_preview()
    logger.info("GoPro setup complete")

    preview_thread = threading.Thread(target=_preview_algorithm, name="preview", daemon=True)
    moment_thread = threading.Thread(target=_moment_detector, name="detector", daemon=True)
    preview_thread.start()
    moment_thread.start()
    logger.info("Started preview and moment detector threads")
    return [preview_thread, moment_thread]


if __name__ == "__main__":
    main()
