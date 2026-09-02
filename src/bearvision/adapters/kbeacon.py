"""BLE tag-scanner adapter for the existing BleBeaconHandler."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import ctypes
from types import SimpleNamespace
from typing import Any

from bearvision.contracts import TagObservation, Vector3

from ._errors import translated_error


STANDARD_GRAVITY_MPS2 = 9.80665
KSENSOR_TYPE = 0x21
SENSOR_MASK_VOLTAGE = 0x1
SENSOR_MASK_ACC_AXIS = 0x8
BEAR_TAG_NAME_PREFIX = "bear_tag"


class BleakKBeaconSource:
    """Minimal packaged KBeacon advertisement source used in production."""

    def __init__(self) -> None:
        self.advertisement_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    @staticmethod
    def _decode(data: bytes) -> tuple[int | None, Any | None]:
        if len(data) < 3 or data[0] != KSENSOR_TYPE:
            return None, None
        mask = int.from_bytes(data[1:3], byteorder="big")
        voltage = int.from_bytes(data[3:5], byteorder="big") if mask & SENSOR_MASK_VOLTAGE else None
        acceleration = None
        if mask & SENSOR_MASK_ACC_AXIS and len(data) >= 13:
            acceleration = SimpleNamespace(
                x=ctypes.c_int16(int.from_bytes(data[7:9], "big")).value / 1000.0,
                y=ctypes.c_int16(int.from_bytes(data[9:11], "big")).value / 1000.0,
                z=ctypes.c_int16(int.from_bytes(data[11:13], "big")).value / 1000.0,
            )
        return voltage, acceleration

    async def look_for_advertisements(
        self,
        timeout: float = 0.0,
        *,
        stop_timeout: float | None = None,
    ) -> None:
        if stop_timeout is not None and stop_timeout <= 0:
            raise ValueError("BLE scanner stop timeout must be positive")
        try:
            from bleak import BleakScanner
        except ImportError as exc:  # pragma: no cover - production dependency
            raise RuntimeError("bleak is required for KBeacon scanning") from exc

        async def callback(device: Any, advertisement: Any) -> None:
            name = getattr(advertisement, "local_name", None) or device.name
            if not name or not name.startswith(BEAR_TAG_NAME_PREFIX):
                return
            for data in advertisement.service_data.values():
                voltage, acceleration = self._decode(bytes(data))
                if acceleration is not None:
                    await self.advertisement_queue.put(
                        {
                            "tag_id": name,
                            "address": device.address,
                            "name": name,
                            "rssi": advertisement.rssi,
                            "batteryLevel": voltage,
                            "acc_sensor": acceleration,
                        }
                    )

        scanner = BleakScanner(detection_callback=callback)
        start_attempted = False
        try:
            start_attempted = True
            await scanner.start()
            if timeout == 0.0:
                await asyncio.Future()
            else:
                await asyncio.sleep(timeout)
        finally:
            if start_attempted:
                if stop_timeout is None:
                    await scanner.stop()
                else:
                    await asyncio.wait_for(scanner.stop(), timeout=stop_timeout)


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
        return TagObservation(
            tag_id=str(raw.get("tag_id", raw["address"])),
            observed_at_utc=self.clock.utc_now(),
            observed_at_monotonic_s=self.clock.monotonic(),
            rssi_dbm=int(raw["rssi"]),
            acceleration_mps2=Vector3(
                x=float(acceleration.x) * STANDARD_GRAVITY_MPS2,
                y=float(acceleration.y) * STANDARD_GRAVITY_MPS2,
                z=float(acceleration.z) * STANDARD_GRAVITY_MPS2,
            ),
            battery_voltage_mv=int(battery) if battery is not None else None,
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
