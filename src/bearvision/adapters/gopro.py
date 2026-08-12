"""Camera port adapter for the existing GoProController."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from bearvision.contracts import CaptureRequest, MediaAsset
from bearvision.ports import CapturedMedia, InvalidComponentData

from ._errors import translated_error


class GoProCameraAdapter:
    def __init__(
        self,
        controller: Any,
        clock: Any,
        capture_dir: str | Path,
        *,
        hindsight_enabled: bool = True,
    ) -> None:
        self.controller = controller
        self.clock = clock
        self.capture_dir = Path(capture_dir)
        self.hindsight_enabled = hindsight_enabled
        self._captures: dict[str, CapturedMedia] = {}

    async def connect(self) -> None:
        try:
            await asyncio.to_thread(self.controller.connect)
        except Exception as exc:
            raise translated_error(exc, "connect GoPro") from exc

    async def disconnect(self) -> None:
        try:
            await asyncio.to_thread(self.controller.disconnect)
        except Exception as exc:
            raise translated_error(exc, "disconnect GoPro") from exc

    async def start_preview(self) -> str:
        try:
            return await asyncio.to_thread(self.controller.start_preview)
        except Exception as exc:
            raise translated_error(exc, "start GoPro preview") from exc

    async def stop_preview(self) -> None:
        try:
            await asyncio.to_thread(self.controller.stop_preview)
        except Exception as exc:
            raise translated_error(exc, "stop GoPro preview") from exc

    async def capture(self, request: CaptureRequest) -> CapturedMedia:
        cached = self._captures.get(request.request_id)
        if cached is not None:
            return cached
        try:
            before = set(await asyncio.to_thread(self.controller.list_videos))
            if self.hindsight_enabled:
                await asyncio.to_thread(self.controller.start_hindsight_clip, request.post_roll_s)
            else:
                await asyncio.to_thread(self.controller.start_recording)
                await self.clock.sleep(request.post_roll_s)
                await asyncio.to_thread(self.controller.stop_recording)
            after = list(await asyncio.to_thread(self.controller.list_videos))
            new_files = [name for name in after if name not in before]
            if not new_files:
                raise InvalidComponentData("GoPro capture produced no new media file")
            camera_file = new_files[-1]
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            destination = self.capture_dir / f"{request.request_id}-{Path(camera_file).name}"
            local_path = Path(
                await asyncio.to_thread(self.controller.download_file, camera_file, str(destination))
            )
            if not local_path.exists():
                raise InvalidComponentData(f"GoPro download is missing: {local_path}")
            media = CapturedMedia(
                asset=MediaAsset(
                    asset_id=f"asset-{request.request_id}",
                    filename=local_path.name,
                    content_type="video/mp4",
                    size_bytes=local_path.stat().st_size,
                    created_at_utc=self.clock.utc_now(),
                ),
                local_path=local_path,
            )
            self._captures[request.request_id] = media
            return media
        except Exception as exc:
            raise translated_error(exc, "capture GoPro media") from exc
