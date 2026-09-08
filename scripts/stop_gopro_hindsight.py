"""Put a USB-connected GoPro in the ready-for-maintenance state."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable
import sys
from typing import Any

from open_gopro import WiredGoPro
from open_gopro.domain.exceptions import FailedToFindDevice
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
    allow_no_gopro: bool = False,
    discover: Callable[[str, int], Awaitable[Any]] = find_first_ip_addr,
    camera_factory: Callable[..., Any] = WiredGoPro,
) -> str | None:
    """Make the camera ready for maintenance without starting USB control."""

    try:
        discovered = await discover(GOPRO_WEB_SERVICE, timeout_s)
    except FailedToFindDevice:
        if allow_no_gopro:
            return None
        raise
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
    parser.add_argument(
        "--allow-no-gopro",
        "--allow_no_gopro",
        action="store_true",
        help="continue when no GoPro is discovered",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    try:
        serial = asyncio.run(
            set_gopro_ready_for_maintenance(
                timeout_s=args.timeout,
                allow_no_gopro=args.allow_no_gopro,
            )
        )
    except Exception as exc:
        print(
            f"[BearVision redeploy] ERROR: could not verify GoPro HindSight OFF: {exc}",
            file=sys.stderr,
        )
        return 1

    if serial is None:
        print(
            "[BearVision redeploy] No GoPro detected; continuing because "
            "--allow-no-gopro was specified"
        )
        return 0

    print(
        f"[BearVision redeploy] GoPro {serial} is ready for maintenance "
        "(HindSight OFF)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
