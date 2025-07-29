import logging
import threading
import time


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
    """Start preview and moment detection threads."""
    logging.basicConfig(level=logging.INFO)
    preview_thread = threading.Thread(target=_preview_algorithm, name="preview", daemon=True)
    moment_thread = threading.Thread(target=_moment_detector, name="detector", daemon=True)
    preview_thread.start()
    moment_thread.start()
    logger.info("Started preview and moment detector threads")
    return [preview_thread, moment_thread]


if __name__ == "__main__":
    main()
