"""BLE tag-scanner adapter for the existing BleBeaconHandler."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from bearvision.contracts import TagObservation, Vector3

from ._errors import translated_error


STANDARD_GRAVITY_MPS2 = 9.80665


class KBeaconTagScannerAdapter:
    def __init__(
        self,
        handler: Any,
        clock: Any,
        *,
        manage_scan: bool = True,
        maximum_observations: int | None = None,
    ) -> None:
        self.handler = handler
        self.clock = clock
        self.manage_scan = manage_scan
        self.maximum_observations = maximum_observations

    def _convert(self, raw: dict[str, Any]) -> TagObservation:
        acceleration = raw["acc_sensor"]
        battery = raw.get("batteryLevel")
        if battery is not None and not 0 <= float(battery) <= 100:
            battery = None
        return TagObservation(
            tag_id=str(raw["address"]),
            observed_at_utc=self.clock.utc_now(),
            observed_at_monotonic_s=self.clock.monotonic(),
            rssi_dbm=int(raw["rssi"]),
            acceleration_mps2=Vector3(
                x=float(acceleration.x) * STANDARD_GRAVITY_MPS2,
                y=float(acceleration.y) * STANDARD_GRAVITY_MPS2,
                z=float(acceleration.z) * STANDARD_GRAVITY_MPS2,
            ),
            battery_percent=float(battery) if battery is not None else None,
        )

    async def observations(self):
        scan_task = None
        if self.manage_scan:
            scan_task = asyncio.create_task(self.handler.look_for_advertisements(timeout=0.0))
        count = 0
        try:
            while self.maximum_observations is None or count < self.maximum_observations:
                raw = await self.handler.advertisement_queue.get()
                try:
                    yield self._convert(raw)
                    count += 1
                except Exception as exc:
                    raise translated_error(exc, "decode KBeacon observation") from exc
                finally:
                    self.handler.advertisement_queue.task_done()
        finally:
            if scan_task is not None:
                scan_task.cancel()
                with suppress(asyncio.CancelledError):
                    await scan_task
