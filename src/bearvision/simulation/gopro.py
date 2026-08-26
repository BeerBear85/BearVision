"""Disk-backed GoPro controller emulator used below the production camera adapter."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path, PurePosixPath
import shutil
import threading
import time
from typing import Any, Callable

from bearvision.ports import VideoClipper


class SimulatedGoProController:
    """Model the stateful subset of ``GoProController`` used by BearVision.

    The preview video acts as the camera's rolling sensor timeline. Recordings
    are materialized on a simulated SD card and subsequently downloaded through
    the same ``GoProCameraAdapter`` used for physical hardware.
    """

    def __init__(
        self,
        *,
        root_dir: str | Path,
        preview_source: str | Path,
        clock: Any,
        clipper: VideoClipper,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.preview_source = Path(preview_source).resolve()
        self.clock = clock
        self.clipper = clipper
        self.sleep = sleep or time.sleep
        self.media_dir = self.root_dir / "100GOPRO"
        self.state_path = self.root_dir / "camera-state.json"
        self.connected = False
        self.previewing = False
        self.encoding = False
        self.hindsight_duration_s = 0
        self._hindsight_enabled_at_s: float | None = None
        self._capture_start_s: float | None = None
        self._media_windows: dict[str, tuple[float, float]] = {}
        self._lock = threading.RLock()
        self._load_persistent_state()

    def connect(self) -> None:
        with self._lock:
            if not self.preview_source.is_file():
                raise ConnectionError(
                    f"simulated GoPro preview source is missing: {self.preview_source}"
                )
            self.media_dir.mkdir(parents=True, exist_ok=True)
            self.connected = True

    def disconnect(self) -> None:
        with self._lock:
            if self.encoding:
                raise RuntimeError("cannot disconnect simulated GoPro while encoding")
            self.previewing = False
            self.connected = False

    def configure(self) -> None:
        if not self.enable_hindsight(15):
            raise RuntimeError("failed to configure simulated GoPro")

    def enable_hindsight(self, duration_s: int = 15) -> bool:
        if duration_s not in {15, 30}:
            raise ValueError("HindSight duration must be 15 or 30 seconds")
        with self._lock:
            self._require_connected()
            self.hindsight_duration_s = duration_s
            self._hindsight_enabled_at_s = float(self.clock.monotonic())
            self._save_persistent_state()
        return True

    def startHindsightMode(self) -> bool:
        return self.enable_hindsight(15)

    def disableHindsightMode(self) -> None:
        with self._lock:
            self._require_connected()
            self.hindsight_duration_s = 0
            self._hindsight_enabled_at_s = None
            self._save_persistent_state()

    def start_preview(self, port: int = 8554) -> str:
        del port  # The emulator exposes the source file as its preview transport.
        with self._lock:
            self._require_connected()
            self.previewing = True
            return str(self.preview_source)

    def stop_preview(self) -> None:
        with self._lock:
            self._require_connected()
            self.previewing = False

    def start_recording(self) -> None:
        with self._lock:
            self._require_connected()
            if self.encoding:
                raise RuntimeError("simulated GoPro is already encoding")
            triggered_at_s = float(self.clock.monotonic())
            earliest_available_s = (
                self._hindsight_enabled_at_s
                if self._hindsight_enabled_at_s is not None
                else triggered_at_s
            )
            self._capture_start_s = max(
                earliest_available_s,
                triggered_at_s - float(self.hindsight_duration_s),
            )
            self.encoding = True

    def stop_recording(self) -> None:
        with self._lock:
            self._require_connected()
            if not self.encoding or self._capture_start_s is None:
                raise RuntimeError("simulated GoPro is not encoding")
            capture_start_s = self._capture_start_s
            capture_end_s = float(self.clock.monotonic())
            self.encoding = False
            self._capture_start_s = None
        duration_s = capture_end_s - capture_start_s
        if duration_s <= 0:
            raise RuntimeError("simulated GoPro recording has no duration")
        destination = self.media_dir / self._next_media_filename()
        camera_file = destination.relative_to(self.root_dir).as_posix()
        try:
            asyncio.run(
                self.clipper.extract(
                    self.preview_source,
                    destination,
                    start_s=capture_start_s,
                    duration_s=duration_s,
                )
            )
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        with self._lock:
            self._media_windows[camera_file] = (capture_start_s, capture_end_s)

    def start_hindsight_clip(self, duration: float = 1.0) -> None:
        """Compatibility helper for legacy callers outside the camera adapter."""
        self.start_recording()
        self.sleep(duration)
        self.stop_recording()

    def list_videos(self) -> list[str]:
        with self._lock:
            self._require_connected()
            if not self.media_dir.is_dir():
                return []
            return [
                path.relative_to(self.root_dir).as_posix()
                for path in sorted(self.media_dir.glob("*.MP4"))
            ]

    def download_file(self, camera_file: str, local_path: str) -> Path:
        with self._lock:
            self._require_connected()
            source = self._camera_path(camera_file)
            if not source.is_file():
                raise FileNotFoundError(camera_file)
        destination = Path(local_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    def get_media_capture_window(self, camera_file: str) -> tuple[float, float]:
        """Return exact simulator timing for media created in this process."""

        with self._lock:
            self._require_connected()
            try:
                return self._media_windows[camera_file]
            except KeyError as exc:
                raise FileNotFoundError(camera_file) from exc

    def delete_file(self, camera_file: str) -> None:
        with self._lock:
            self._require_connected()
            source = self._camera_path(camera_file)
            if not source.is_file():
                raise FileNotFoundError(camera_file)
            source.unlink()

    def delete_all_files(self) -> None:
        with self._lock:
            self._require_connected()
            for path in self.media_dir.glob("*.MP4"):
                path.unlink()

    def get_camera_status(self) -> dict[str, dict[str, int | bool]]:
        with self._lock:
            self._require_connected()
            return {
                "status": {
                    "encoding": self.encoding,
                    "preview_stream": self.previewing,
                    "preview_stream_available": True,
                },
                "settings": {"hindsight_duration_s": self.hindsight_duration_s},
            }

    def _camera_path(self, camera_file: str) -> Path:
        relative = Path(*PurePosixPath(camera_file.replace("\\", "/")).parts)
        candidate = (self.root_dir / relative).resolve()
        if self.root_dir != candidate and self.root_dir not in candidate.parents:
            raise ValueError("camera media path escapes simulated SD card")
        return candidate

    def _next_media_filename(self) -> str:
        existing = {path.name.upper() for path in self.media_dir.glob("GX??????.MP4")}
        for number in range(1, 10_000):
            filename = f"GX01{number:04d}.MP4"
            if filename not in existing:
                return filename
        raise RuntimeError("simulated GoPro media namespace is exhausted")

    def _require_connected(self) -> None:
        if not self.connected:
            raise ConnectionError("simulated GoPro is disconnected")

    def _load_persistent_state(self) -> None:
        if not self.state_path.is_file():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            duration_s = int(data.get("hindsight_duration_s", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid simulated GoPro state: {self.state_path}") from exc
        if duration_s not in {0, 15, 30}:
            raise ValueError("persisted HindSight duration must be 0, 15 or 30 seconds")
        self.hindsight_duration_s = duration_s

    def _save_persistent_state(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        partial = self.state_path.with_suffix(".partial.json")
        partial.write_text(
            json.dumps(
                {"hindsight_duration_s": self.hindsight_duration_s},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        partial.replace(self.state_path)
