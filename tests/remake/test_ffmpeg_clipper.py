import asyncio
import json
from pathlib import Path
import subprocess

import pytest

from bearvision.adapters import FfmpegVideoClipper
from bearvision.config.models import ClipExtractionConfig
from bearvision.ports import InvalidComponentData


class FakeMediaCommands:
    def __init__(self, *, fail_ffmpeg: bool = False, output_duration_s: float = 5.0) -> None:
        self.fail_ffmpeg = fail_ffmpeg
        self.output_duration_s = output_duration_s
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if command[0] == "ffmpeg-test":
            partial = Path(command[-1])
            partial.write_bytes(b"extracted-video")
            return subprocess.CompletedProcess(
                command,
                1 if self.fail_ffmpeg else 0,
                "",
                "injected failure" if self.fail_ffmpeg else "",
            )
        path = Path(command[-1])
        duration = 15.296 if path.name == "source.mp4" else self.output_duration_s
        metadata = {
            "format": {"duration": str(duration)},
            "streams": [
                {"codec_type": "video", "width": 320, "height": 180},
                {"codec_type": "audio"},
            ],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(metadata), "")


def build_clipper(commands: FakeMediaCommands) -> FfmpegVideoClipper:
    return FfmpegVideoClipper(
        ClipExtractionConfig(),
        ffmpeg_path="ffmpeg-test",
        ffprobe_path="ffprobe-test",
        run_command=commands,
    )


def test_clipper_uses_accurate_seek_and_atomically_publishes(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original-video")
    destination = tmp_path / "captures" / "capture-180.mp4"
    commands = FakeMediaCommands()

    result = asyncio.run(
        build_clipper(commands).extract(
            source,
            destination,
            start_s=6.006,
            duration_s=5.0,
        )
    )

    ffmpeg = next(command for command in commands.commands if command[0] == "ffmpeg-test")
    assert ffmpeg.index("-ss") > ffmpeg.index("-i")
    assert ffmpeg[ffmpeg.index("-ss") + 1] == "6.006000000"
    assert result.path == destination.resolve()
    assert result.duration_s == 5.0
    assert result.has_audio is True
    assert destination.read_bytes() == b"extracted-video"
    assert source.read_bytes() == b"original-video"
    assert not list(tmp_path.rglob("*.partial.mp4"))


def test_clipper_removes_partial_output_after_failure(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original-video")
    commands = FakeMediaCommands(fail_ffmpeg=True)

    with pytest.raises(InvalidComponentData, match="injected failure"):
        asyncio.run(
            build_clipper(commands).extract(
                source,
                tmp_path / "capture.mp4",
                start_s=1,
                duration_s=5,
            )
        )

    assert not list(tmp_path.glob("*.partial.mp4"))


def test_existing_valid_clip_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original-video")
    destination = tmp_path / "capture.mp4"
    destination.write_bytes(b"existing-clip")
    commands = FakeMediaCommands()

    result = asyncio.run(
        build_clipper(commands).extract(
            source,
            destination,
            start_s=6,
            duration_s=5,
        )
    )

    assert result.path == destination.resolve()
    assert not any(command[0] == "ffmpeg-test" for command in commands.commands)
