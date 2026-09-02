"""Asynchronous Open GoPro integration used by the BearVision 3 camera adapter."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from open_gopro import WiredGoPro
from open_gopro.models import proto
from open_gopro.models.constants import SettingId, StatusId, constants, settings


logger = logging.getLogger(__name__)


class AsyncGoProController:
    """Expose the Open GoPro SDK without crossing threads or event loops."""

    def __init__(self, target: str | None = None, *, gopro: Any | None = None) -> None:
        self._gopro = gopro if gopro is not None else WiredGoPro(target=target)

    async def connect(self) -> None:
        await self._gopro.open()
        video_group = proto.EnumPresetGroup.PRESET_GROUP_ID_VIDEO
        if await self._current_preset_group() == video_group:
            return
        response = await self._gopro.http_command.load_preset_group(
            group=video_group
        )
        if (
            hasattr(response, "ok")
            and not response.ok
            and await self._current_preset_group() != video_group
        ):
            raise RuntimeError("GoPro rejected the Video preset group")

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
        return (
            f"udp://@0.0.0.0:{port}"
            "?fifo_size=65536&overrun_nonfatal=1"
        )

    async def stop_preview(self) -> None:
        await self._gopro.http_command.set_preview_stream(
            mode=constants.Toggle.DISABLE
        )

    async def _current_preset_group(self) -> Any | None:
        try:
            response = await self._gopro.http_command.get_camera_state()
        except Exception as exc:
            logger.warning("Could not read GoPro preset group: %s", exc)
            return None
        if hasattr(response, "ok") and not response.ok:
            return None
        if not isinstance(response.data, dict):
            return None
        return response.data.get(StatusId.PRESET_GROUP)

    async def _current_hindsight(self) -> Any | None:
        try:
            response = await self._gopro.http_command.get_camera_state()
        except Exception as exc:
            logger.warning("Could not read GoPro state before setting HindSight: %s", exc)
            return None
        if hasattr(response, "ok") and not response.ok:
            return None
        if not isinstance(response.data, dict):
            return None
        return response.data.get(SettingId.HINDSIGHT)

    async def enable_hindsight(self, duration_s: int = 15) -> bool:
        values = {
            15: settings.Hindsight.NUM_15_SECONDS,
            30: settings.Hindsight.NUM_30_SECONDS,
        }
        try:
            value = values[duration_s]
        except KeyError as exc:
            raise ValueError("HindSight duration must be 15 or 30 seconds") from exc

        if await self._current_hindsight() == value:
            return True

        response = await self._gopro.http_setting.hindsight.set(value)
        if not hasattr(response, "ok") or response.ok:
            return True

        return await self._current_hindsight() == value

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
