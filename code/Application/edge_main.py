"""Edge device entry point running lightweight detection alongside the GoPro."""

import logging
import threading
import asyncio
import time
from pathlib import Path
import sys

MODULE_DIR = Path(__file__).resolve().parents[1] / "modules"
sys.path.append(str(MODULE_DIR))

from ConfigurationHandler import ConfigurationHandler
from GoProController import GoProController


logger = logging.getLogger(__name__)


async def _preview_algorithm(event: asyncio.Event) -> None:
    """Analyse the preview stream and set ``event`` when motion is found.

    Parameters
    ----------
    event : asyncio.Event
        Synchronisation primitive used to notify other tasks of motion.

    Returns
    -------
    None
        The function runs forever and communicates via ``event``.
    """
    counter = 0
    while True:
        await asyncio.sleep(0.1)
        # Dummy loop increments a counter to simulate motion detection cadence.
        counter += 1
        if counter % 50 == 0:
            logger.debug("Motion detected in preview")
            event.set()


async def _hindsight_trigger(event: asyncio.Event, gopro: GoProController | None) -> None:
    """Wait for motion and optionally trigger a HindSight clip on the GoPro.

    Parameters
    ----------
    event : asyncio.Event
        Event signalled by ``_preview_algorithm`` when motion occurs.
    gopro : GoProController | None
        Connected GoPro controller or ``None`` when running without a camera.
    """
    while True:
        await event.wait()
        event.clear()
        logger.info("HindSight clip triggered")
        if gopro:
            gopro.start_hindsight_clip()


def _run_async(event: asyncio.Event, gopro: GoProController | None) -> None:
    """Run preview and HindSight tasks in the same event loop."""

    async def _runner() -> None:
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
    """Configure environment, optionally setup GoPro and start worker threads.

    Parameters
    ----------
    gopro : GoProController | None, optional
        Existing GoPro controller. If ``None`` a new one is created.

    Returns
    -------
    list[threading.Thread]
        Started threads handling preview processing and moment detection.
    """
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
    # Separate threads allow the async event loop and traditional blocking code to coexist.
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
