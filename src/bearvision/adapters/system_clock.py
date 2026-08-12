"""Production clock using UTC wall time and process monotonic time."""

import asyncio
from datetime import datetime, timezone
import time


class SystemClock:
    def utc_now(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, delay_s: float) -> None:
        if delay_s < 0:
            raise ValueError("delay_s must not be negative")
        await asyncio.sleep(delay_s)
