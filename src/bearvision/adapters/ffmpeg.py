"""Frame-accurate local MP4 extraction through FFmpeg on the Edge computer."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Callable

from bearvision.config.models import ClipExtractionConfig
from bearvision.ports import ComponentUnavailable, ExtractedClip, InvalidComponentData


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


class FfmpegVideoClipper:
    """Re-encode an exact media window and atomically publish the result."""

    def __init__(
        self,
        config: ClipExtractionConfig,
        *,
        ffmpeg_path: str | Path | None = None,
        ffprobe_path: str | Path | None = None,
        run_command: CommandRunner | None = None,
        duration_tolerance_s: float = 0.1,
    ) -> None:
        if duration_tolerance_s < 0:
            raise ValueError("duration tolerance must not be negative")
        self.config = config
        self.ffmpeg_path = self._resolve_executable(
            ffmpeg_path, "BEARVISION_FFMPEG", "ffmpeg"
        )
        self.ffprobe_path = self._resolve_executable(
            ffprobe_path, "BEARVISION_FFPROBE", "ffprobe"
        )
        self.run_command = run_command or self._run_command
        self.duration_tolerance_s = duration_tolerance_s

    @staticmethod
    def _resolve_executable(
        configured: str | Path | None,
        environment_name: str,
        default_name: str,
    ) -> str:
        candidate = str(configured) if configured is not None else os.getenv(environment_name)
        if candidate:
            return candidate
        return shutil.which(default_name) or default_name

    @staticmethod
    def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            raise ComponentUnavailable(f"media executable is unavailable: {command[0]}") from exc

    async def extract(
        self,
        source: Path,
        destination: Path,
        *,
        start_s: float,
        duration_s: float,
    ) -> ExtractedClip:
        return await asyncio.to_thread(
            self._extract_sync,
            source,
            destination,
            start_s,
            duration_s,
        )

    def _extract_sync(
        self,
        source: Path,
        destination: Path,
        start_s: float,
        duration_s: float,
    ) -> ExtractedClip:
        source = source.resolve()
        destination = destination.resolve()
        if not source.is_file():
            raise InvalidComponentData(f"clip source does not exist: {source}")
        if source == destination:
            raise InvalidComponentData("clip destination must differ from source")
        if start_s < 0 or duration_s <= 0:
            raise InvalidComponentData("clip start and duration are invalid")

        source_info = self._probe(source)
        source_duration = float(source_info["duration_s"])
        if start_s >= source_duration:
            raise InvalidComponentData("clip starts after the source has ended")
        expected_duration = min(duration_s, source_duration - start_s)
        if destination.is_file():
            return self._validated_result(destination, start_s, expected_duration)

        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f"{destination.stem}.partial{destination.suffix}")
        partial.unlink(missing_ok=True)
        command = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ss",
            f"{start_s:.9f}",
            "-t",
            f"{expected_duration:.9f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            self.config.video_codec,
            "-preset",
            self.config.preset,
            "-crf",
            str(self.config.crf),
            "-c:a",
            self.config.audio_codec,
            "-movflags",
            "+faststart",
            str(partial),
        ]
        try:
            completed = self.run_command(command)
            if completed.returncode != 0:
                message = completed.stderr.strip() or "FFmpeg returned a non-zero exit code"
                raise InvalidComponentData(f"clip extraction failed: {message}")
            result = self._validated_result(partial, start_s, expected_duration)
            os.replace(partial, destination)
            return ExtractedClip(
                path=destination,
                start_s=result.start_s,
                duration_s=result.duration_s,
                width_px=result.width_px,
                height_px=result.height_px,
                has_audio=result.has_audio,
            )
        finally:
            partial.unlink(missing_ok=True)

    def _validated_result(
        self,
        path: Path,
        start_s: float,
        expected_duration_s: float,
    ) -> ExtractedClip:
        info = self._probe(path)
        actual_duration = float(info["duration_s"])
        if abs(actual_duration - expected_duration_s) > self.duration_tolerance_s:
            raise InvalidComponentData(
                "extracted clip duration differs from request: "
                f"expected {expected_duration_s:.3f}s, got {actual_duration:.3f}s"
            )
        return ExtractedClip(
            path=path,
            start_s=start_s,
            duration_s=actual_duration,
            width_px=int(info["width_px"]),
            height_px=int(info["height_px"]),
            has_audio=bool(info["has_audio"]),
        )

    def _probe(self, path: Path) -> dict[str, float | int | bool]:
        completed = self.run_command(
            [
                self.ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type,width,height",
                "-of",
                "json",
                str(path),
            ]
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or "FFprobe returned a non-zero exit code"
            raise InvalidComponentData(f"cannot probe media: {message}")
        try:
            data = json.loads(completed.stdout)
            streams = data["streams"]
            video = next(item for item in streams if item["codec_type"] == "video")
            return {
                "duration_s": float(data["format"]["duration"]),
                "width_px": int(video["width"]),
                "height_px": int(video["height"]),
                "has_audio": any(item["codec_type"] == "audio" for item in streams),
            }
        except (KeyError, TypeError, ValueError, StopIteration, json.JSONDecodeError) as exc:
            raise InvalidComponentData("FFprobe returned invalid media metadata") from exc
