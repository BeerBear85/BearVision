import logging
import threading
import asyncio
import time


logger = logging.getLogger(__name__)


async def _preview_algorithm(event: asyncio.Event) -> None:
    """Placeholder preview algorithm using asyncio."""
    while True:
        await asyncio.sleep(0.1)
        # In real code this would analyse the preview stream
        # and set the event when motion is detected.
        # event.set()


async def _hindsight_trigger(event: asyncio.Event) -> None:
    """Wait for the motion event and trigger HindSight recording."""
    while True:
        await event.wait()
        event.clear()
        # Placeholder for GoPro HindSight trigger
        logger.info("HindSight clip triggered")


def _run_async(event: asyncio.Event) -> None:
    """Run preview and HindSight tasks in an event loop."""
    async def _runner() -> None:
        await asyncio.gather(
            _preview_algorithm(event),
            _hindsight_trigger(event),
        )

    asyncio.run(_runner())


def _moment_detector() -> None:
    """Placeholder moment detection running in a thread."""
    while True:
        time.sleep(0.1)


def main() -> list[threading.Thread]:
    """Start preview and moment detection threads."""
    logging.basicConfig(level=logging.INFO)
    motion_event = asyncio.Event()
    preview_thread = threading.Thread(
        target=_run_async,
        args=(motion_event,),
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
