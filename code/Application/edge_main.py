import logging
import threading
import asyncio
import time
from pathlib import Path
    from modules.GoProController import GoProController
sys.path.append(str(MODULE_DIR))

from ConfigurationHandler import ConfigurationHandler


logger = logging.getLogger(__name__)


async def _preview_algorithm(event: asyncio.Event) -> None:
    """Simulate preview processing and emit a motion event."""
    counter = 0
    while True:
        await asyncio.sleep(0.1)
        # In real code this would analyse the preview stream and
        # call ``event.set()`` whenever motion is detected.
        counter += 1
        if counter % 50 == 0:
            logger.debug("Motion detected in preview")
            event.set()


async def _hindsight_trigger(event: asyncio.Event, gopro: GoProController | None) -> None:
    """Wait for the motion event and optionally trigger a HindSight clip."""
    while True:
        await event.wait()
        event.clear()
        logger.info("HindSight clip triggered")
        if gopro:
            gopro.start_hindsight_clip()


def _run_async(event: asyncio.Event, gopro: GoProController | None) -> None:
    """Run preview and HindSight tasks in an event loop."""
    async def _runner() -> None:
        if gopro:
            gopro.start_preview()
        await asyncio.gather(
            _preview_algorithm(event),
            _hindsight_trigger(event, gopro),
        )

    asyncio.run(_runner())


def _moment_detector() -> None:
    """Placeholder moment detection running in a thread."""
    while True:
        time.sleep(0.1)


def main(gopro: GoProController | None = None) -> list[threading.Thread]:
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

    motion_event = asyncio.Event()
    preview_thread = threading.Thread(
        target=_run_async,
        args=(motion_event, gopro),
        name="preview",
        daemon=True,
    )
    moment_thread = threading.Thread(target=_moment_detector, name="detector", daemon=True)
    preview_thread.start()
    moment_thread.start()
    logger.info("Started preview and moment detector threads")
    return [preview_thread, moment_thread]


if __name__ == "__main__":
    main()
