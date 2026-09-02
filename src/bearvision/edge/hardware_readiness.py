"""Bounded, non-destructive physical hardware handshakes for Edge readiness."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Protocol


class PhysicalHandshake(Protocol):
    async def check_camera_preview(self) -> str: ...

    async def check_ble_scanner(self) -> str: ...


class PhysicalReadinessHandshake:
    """Prove the production preview and BLE scan paths through a small interface."""

    def __init__(
        self,
        *,
        camera_factory: Callable[[], Any],
        frame_source_factory: Callable[[], Any],
        ble_source_factory: Callable[[], Any],
        camera_timeout_s: float,
        ble_scan_duration_s: float,
        cleanup_timeout_s: float,
    ) -> None:
        if camera_timeout_s <= 0:
            raise ValueError("camera timeout must be positive")
        if ble_scan_duration_s <= 0:
            raise ValueError("BLE scan duration must be positive")
        if cleanup_timeout_s <= 0:
            raise ValueError("cleanup timeout must be positive")
        self.camera_factory = camera_factory
        self.frame_source_factory = frame_source_factory
        self.ble_source_factory = ble_source_factory
        self.camera_timeout_s = camera_timeout_s
        self.ble_scan_duration_s = ble_scan_duration_s
        self.cleanup_timeout_s = cleanup_timeout_s

    async def check_camera_preview(self) -> str:
        """Connect, receive one preview frame, and release every camera resource."""

        camera = self.camera_factory()
        frames = self.frame_source_factory()
        connection_attempted = False
        preview_attempted = False
        frames_open_attempted = False
        failure: BaseException | None = None
        evidence: str | None = None
        try:
            async with asyncio.timeout(self.camera_timeout_s):
                connection_attempted = True
                await camera.connect()
                preview_attempted = True
                source = await camera.start_preview()
                frames_open_attempted = True
                await frames.open(source)
                frame = await anext(frames.frames())
                evidence = (
                    f"GoPro preview received a {frame.width_px}x{frame.height_px} frame"
                )
        except TimeoutError:
            failure = TimeoutError(
                "GoPro preview handshake did not complete within "
                f"{self.camera_timeout_s:g} seconds"
            )
        except BaseException as exc:
            failure = exc

        cleanup_failure: BaseException | None = None
        cleanup_operations: list[tuple[str, Callable[[], Any]]] = []
        if frames_open_attempted:
            cleanup_operations.append(("closing preview frames", frames.close))
        if preview_attempted:
            cleanup_operations.append(("stopping camera preview", camera.stop_preview))
        if connection_attempted:
            cleanup_operations.append(("disconnecting camera", camera.disconnect))
        for operation_name, operation in cleanup_operations:
            try:
                await asyncio.wait_for(operation(), timeout=self.cleanup_timeout_s)
            except TimeoutError:
                cleanup_failure = cleanup_failure or TimeoutError(
                    "GoPro cleanup did not complete while "
                    f"{operation_name} within {self.cleanup_timeout_s:g} seconds"
                )
            except BaseException as exc:
                cleanup_failure = cleanup_failure or RuntimeError(
                    f"GoPro cleanup failed while {operation_name}: {exc}"
                )

        if failure is not None:
            raise failure
        if cleanup_failure is not None:
            raise cleanup_failure
        if evidence is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("GoPro preview produced no evidence")
        return evidence

    async def check_ble_scanner(self) -> str:
        """Start and stop the production BLE scanner without requiring a tag."""

        source = self.ble_source_factory()
        timeout_s = self.ble_scan_duration_s + self.cleanup_timeout_s
        try:
            await asyncio.wait_for(
                source.look_for_advertisements(
                    timeout=self.ble_scan_duration_s,
                    stop_timeout=self.cleanup_timeout_s,
                ),
                timeout=timeout_s,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                "BLE scanner handshake did not complete within "
                f"{timeout_s:g} seconds"
            ) from exc
        count = int(source.advertisement_queue.qsize())
        noun = "advertisement" if count == 1 else "advertisements"
        return f"BLE scanner completed; {count} BearTag {noun} observed"
