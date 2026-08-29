"""Asynchronous Open GoPro integration used by the BearVision 3 camera adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from open_gopro import WiredGoPro
from open_gopro.models.constants import constants, settings


class AsyncGoProController:
    """Expose the Open GoPro SDK without crossing threads or event loops."""

    def __init__(self, target: str | None = None, *, gopro: Any | None = None) -> None:
        self._gopro = gopro if gopro is not None else WiredGoPro(target=target)

    async def connect(self) -> None:
        await self._gopro.open()

    async def disconnect(self) -> None:
        await self._gopro.close()

    async def list_videos(self) -> list[str]:
        response = await self._gopro.http_command.get_media_list()
        return [item.filename for item in response.data.files]

    async def download_file(self, camera_file: str, local_path: str) -> Path:
        response = await self._gopro.http_command.download_file(
            camera_file=camera_file,
            local_file=Path(local_path),
        )
        return Path(response.data)

    async def start_preview(self, port: int = 8554) -> str:
        try:
            await self._gopro.http_command.set_preview_stream(
                mode=constants.Toggle.DISABLE
            )
            await asyncio.sleep(0.5)
        except Exception:
            # A camera with no active stream may reject the stop command.
            pass
        response = await self._gopro.http_command.set_preview_stream(
            mode=constants.Toggle.ENABLE,
            port=port,
        )
        if hasattr(response, "ok") and not response.ok:
            raise RuntimeError("GoPro rejected preview stream start")
        return f"udp://@0.0.0.0:{port}"

    async def stop_preview(self) -> None:
        await self._gopro.http_command.set_preview_stream(
            mode=constants.Toggle.DISABLE
        )

    async def enable_hindsight(self, duration_s: int = 15) -> bool:
        values = {
            15: settings.Hindsight.NUM_15_SECONDS,
            30: settings.Hindsight.NUM_30_SECONDS,
        }
        try:
            value = values[duration_s]
        except KeyError as exc:
            raise ValueError("HindSight duration must be 15 or 30 seconds") from exc
        response = await self._gopro.http_setting.hindsight.set(value)
        return not hasattr(response, "ok") or bool(response.ok)

    async def disable_hindsight(self) -> None:
        await self._gopro.http_setting.hindsight.set(settings.Hindsight.OFF)

    async def start_recording(self) -> None:
        await self._gopro.http_command.set_shutter(
            shutter=constants.Toggle.ENABLE
        )

    async def stop_recording(self) -> None:
        await self._gopro.http_command.set_shutter(
            shutter=constants.Toggle.DISABLE
        )
