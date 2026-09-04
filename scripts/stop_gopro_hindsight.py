"""Put a USB-connected GoPro in the ready-for-maintenance state."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable
import sys
from typing import Any

from open_gopro import WiredGoPro
from open_gopro.models.constants import SettingId, settings
from open_gopro.network.wifi.mdns_scanner import find_first_ip_addr


GOPRO_WEB_SERVICE = "_gopro-web._tcp.local."


def _hindsight_value(response: Any) -> Any:
    if hasattr(response, "ok") and not response.ok:
        raise RuntimeError("GoPro rejected the camera-state request")
    if not isinstance(response.data, dict):
        raise RuntimeError("GoPro returned an invalid camera state")
    return response.data.get(SettingId.HINDSIGHT)


async def set_gopro_ready_for_maintenance(
    *,
    timeout_s: int = 8,
    discover: Callable[[str, int], Awaitable[Any]] = find_first_ip_addr,
    camera_factory: Callable[..., Any] = WiredGoPro,
) -> str:
    """Make the camera ready for maintenance without starting USB control."""

    discovered = await discover(GOPRO_WEB_SERVICE, timeout_s)
    serial = discovered.name.split(".", 1)[0]
    if not serial:
        raise RuntimeError("GoPro discovery returned no serial number")

    camera = camera_factory(serial=serial)
    off = settings.Hindsight.OFF
    current = _hindsight_value(await camera.http_command.get_camera_state())
    if current != off:
        await camera.http_setting.hindsight.set(off)
        current = _hindsight_value(await camera.http_command.get_camera_state())
    if current != off:
        raise RuntimeError("GoPro did not confirm HindSight OFF")
    return serial


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set the GoPro ready for BearVision maintenance"
    )
    parser.add_argument("--timeout", type=int, default=8)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    try:
        serial = asyncio.run(
            set_gopro_ready_for_maintenance(timeout_s=args.timeout)
        )
    except Exception as exc:
        print(
            f"[BearVision redeploy] ERROR: could not verify GoPro HindSight OFF: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        f"[BearVision redeploy] GoPro {serial} is ready for maintenance "
        "(HindSight OFF)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
