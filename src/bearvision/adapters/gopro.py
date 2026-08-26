"""Camera port adapter for the existing GoProController."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from bearvision.contracts import CaptureRequest, MediaAsset
from bearvision.ports import (
    CapturedClip,
    CapturedMedia,
    CaptureWindow,
    CaptureWindowBasis,
    CaptureWindowPrecision,
    InvalidComponentData,
    MediaProbe,
    requested_capture_window,
)

from ._errors import translated_error


class GoProCameraAdapter:
    def __init__(
        self,
        controller: Any,
        clock: Any,
        capture_dir: str | Path,
        *,
        hindsight_enabled: bool = True,
        hindsight_duration_s: int = 15,
        media_probe: MediaProbe | None = None,
        duration_tolerance_s: float = 0.1,
    ) -> None:
        if hindsight_duration_s not in {15, 30}:
            raise ValueError("HindSight duration must be 15 or 30 seconds")
        if duration_tolerance_s < 0:
            raise ValueError("capture duration tolerance must not be negative")
        self.controller = controller
        self.clock = clock
        self.capture_dir = Path(capture_dir)
        self.hindsight_enabled = hindsight_enabled
        self.hindsight_duration_s = hindsight_duration_s
        self.media_probe = media_probe
        self.duration_tolerance_s = duration_tolerance_s
        self._available_since_monotonic_s: float | None = None
        self._captures: dict[str, CapturedClip] = {}

    async def connect(self) -> None:
        connected = False
        try:
            await asyncio.to_thread(self.controller.connect)
            connected = True
            if self.hindsight_enabled:
                enabled = await asyncio.to_thread(
                    self.controller.enable_hindsight, self.hindsight_duration_s
                )
                if enabled is False:
                    raise RuntimeError("GoPro did not enable HindSight")
            else:
                await asyncio.to_thread(self.controller.disableHindsightMode)
            self._available_since_monotonic_s = self.clock.monotonic()
        except Exception as exc:
            if connected:
                try:
                    await asyncio.to_thread(self.controller.disconnect)
                except Exception:
                    pass
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

    async def capture(self, request: CaptureRequest) -> CapturedClip:
        cached = self._captures.get(request.request_id)
        if cached is not None:
            return cached
        try:
            configured_pre_roll_s = (
                float(self.hindsight_duration_s) if self.hindsight_enabled else 0.0
            )
            if abs(request.pre_roll_s - configured_pre_roll_s) > 1e-9:
                raise InvalidComponentData(
                    "capture pre-roll does not match configured GoPro HindSight: "
                    f"requested {request.pre_roll_s:.3f}s, configured "
                    f"{configured_pre_roll_s:.3f}s"
                )
            if self._available_since_monotonic_s is None and request.pre_roll_s > 0:
                raise InvalidComponentData("GoPro must be connected before a capture with pre-roll")
            earliest_available_s = (
                self._available_since_monotonic_s
                if self._available_since_monotonic_s is not None
                else request.requested_at_monotonic_s
            )
            requested_window = requested_capture_window(
                request,
                earliest_available_monotonic_s=earliest_available_s,
            )
            before = set(await asyncio.to_thread(self.controller.list_videos))
            recording_started = False
            start_command_before_s = self.clock.monotonic()
            start_command_after_s = start_command_before_s
            stop_command_before_s = start_command_before_s
            stop_command_after_s = start_command_before_s
            try:
                await asyncio.to_thread(self.controller.start_recording)
                recording_started = True
                start_command_after_s = self.clock.monotonic()
                await self.clock.sleep(request.post_roll_s)
            finally:
                if recording_started:
                    stop_command_before_s = self.clock.monotonic()
                    await asyncio.to_thread(self.controller.stop_recording)
                    stop_command_after_s = self.clock.monotonic()
            after = list(await asyncio.to_thread(self.controller.list_videos))
            new_files = [name for name in after if name not in before]
            if not new_files:
                raise InvalidComponentData("GoPro capture produced no new media file")
            camera_file = new_files[-1]
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            destination = self.capture_dir / f"{request.request_id}-{Path(camera_file).name}"
            local_path = Path(
                await asyncio.to_thread(
                    self.controller.download_file, camera_file, str(destination)
                )
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
            actual_window = await self._actual_window(
                camera_file,
                local_path,
                earliest_available_s=earliest_available_s,
                start_command_before_s=start_command_before_s,
                start_command_after_s=start_command_after_s,
                stop_command_before_s=stop_command_before_s,
                stop_command_after_s=stop_command_after_s,
            )
            if not (
                actual_window.start_monotonic_s - self.duration_tolerance_s
                <= request.requested_at_monotonic_s
                <= actual_window.end_monotonic_s + self.duration_tolerance_s
            ):
                raise InvalidComponentData(
                    "GoPro raw clip does not contain the detection timestamp"
                )
            capture = CapturedClip(
                request_id=request.request_id,
                media=media,
                requested_window=requested_window,
                actual_window=actual_window,
            )
            self._captures[request.request_id] = capture
            return capture
        except Exception as exc:
            raise translated_error(exc, "capture GoPro media") from exc

    async def _actual_window(
        self,
        camera_file: str,
        local_path: Path,
        *,
        earliest_available_s: float,
        start_command_before_s: float,
        start_command_after_s: float,
        stop_command_before_s: float,
        stop_command_after_s: float,
    ) -> CaptureWindow:
        exact_window_reader = getattr(self.controller, "get_media_capture_window", None)
        if callable(exact_window_reader):
            start_s, end_s = await asyncio.to_thread(exact_window_reader, camera_file)
            window = CaptureWindow(
                start_monotonic_s=float(start_s),
                end_monotonic_s=float(end_s),
                precision=CaptureWindowPrecision.EXACT,
                basis=CaptureWindowBasis.SIMULATED_MEDIA_TIMELINE,
            )
            if self.media_probe is not None:
                duration_s = await self.media_probe.duration(local_path)
                self._validate_media_duration(duration_s, window.duration_s, window.duration_s)
            return window

        available_before_s = min(
            float(self.hindsight_duration_s) if self.hindsight_enabled else 0.0,
            max(0.0, start_command_before_s - earliest_available_s),
        )
        available_after_s = min(
            float(self.hindsight_duration_s) if self.hindsight_enabled else 0.0,
            max(0.0, start_command_after_s - earliest_available_s),
        )
        minimum_duration_s = available_before_s + max(
            0.0, stop_command_before_s - start_command_after_s
        )
        maximum_duration_s = available_after_s + max(
            0.0, stop_command_after_s - start_command_before_s
        )
        stop_estimate_s = (stop_command_before_s + stop_command_after_s) / 2
        if self.media_probe is not None:
            duration_s = await self.media_probe.duration(local_path)
            self._validate_media_duration(duration_s, minimum_duration_s, maximum_duration_s)
            return CaptureWindow(
                start_monotonic_s=max(0.0, stop_estimate_s - duration_s),
                end_monotonic_s=stop_estimate_s,
                precision=CaptureWindowPrecision.ESTIMATED,
                basis=CaptureWindowBasis.CAMERA_COMMAND_TIMING_AND_MEDIA_DURATION,
            )

        start_estimate_s = (start_command_before_s + start_command_after_s) / 2
        available_estimate_s = min(
            float(self.hindsight_duration_s) if self.hindsight_enabled else 0.0,
            max(0.0, start_estimate_s - earliest_available_s),
        )
        return CaptureWindow(
            start_monotonic_s=max(0.0, start_estimate_s - available_estimate_s),
            end_monotonic_s=stop_estimate_s,
            precision=CaptureWindowPrecision.ESTIMATED,
            basis=CaptureWindowBasis.CAMERA_COMMAND_TIMING,
        )

    def _validate_media_duration(
        self,
        actual_duration_s: float,
        minimum_duration_s: float,
        maximum_duration_s: float,
    ) -> None:
        if actual_duration_s <= 0:
            raise InvalidComponentData("GoPro raw clip duration must be positive")
        if actual_duration_s < minimum_duration_s - self.duration_tolerance_s or (
            actual_duration_s > maximum_duration_s + self.duration_tolerance_s
        ):
            raise InvalidComponentData(
                "GoPro raw clip duration is outside the camera-command bounds: "
                f"expected {minimum_duration_s:.3f}-{maximum_duration_s:.3f}s, "
                f"got {actual_duration_s:.3f}s"
            )

    @property
    def captures(self) -> dict[str, CapturedClip]:
        return dict(self._captures)
