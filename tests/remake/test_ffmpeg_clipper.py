import asyncio
import json
from pathlib import Path
import subprocess

import pytest

from bearvision.adapters import FfmpegVideoClipper
from bearvision.config.models import ClipExtractionConfig
from bearvision.ports import ComponentUnavailable, InvalidComponentData


class FakeMediaCommands:
    def __init__(
        self,
        *,
        fail_ffmpeg: bool = False,
        fail_probe: bool = False,
        invalid_probe_json: bool = False,
        output_duration_s: float = 5.0,
        has_audio: bool = True,
    ) -> None:
        self.fail_ffmpeg = fail_ffmpeg
        self.fail_probe = fail_probe
        self.invalid_probe_json = invalid_probe_json
        self.output_duration_s = output_duration_s
        self.has_audio = has_audio
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
        if self.fail_probe:
            return subprocess.CompletedProcess(command, 1, "", "probe failure")
        if self.invalid_probe_json:
            return subprocess.CompletedProcess(command, 0, "not json", "")
        duration = 15.296 if path.name == "source.mp4" else self.output_duration_s
        streams = [{"codec_type": "video", "width": 320, "height": 180}]
        if self.has_audio:
            streams.append({"codec_type": "audio"})
        metadata = {
            "format": {"duration": str(duration)},
            "streams": streams,
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


def test_clip_is_clamped_to_remaining_source_and_supports_video_without_audio(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original-video")
    commands = FakeMediaCommands(output_duration_s=1.296, has_audio=False)

    result = asyncio.run(
        build_clipper(commands).extract(
            source,
            tmp_path / "capture.mp4",
            start_s=14,
            duration_s=5,
        )
    )

    assert result.duration_s == pytest.approx(1.296)
    assert result.has_audio is False
    ffmpeg = next(command for command in commands.commands if command[0] == "ffmpeg-test")
    assert ffmpeg[ffmpeg.index("-t") + 1] == "1.296000000"


@pytest.mark.parametrize(
    ("source_exists", "same_destination", "start_s", "duration_s", "message"),
    [
        (False, False, 0, 5, "source does not exist"),
        (True, True, 0, 5, "destination must differ"),
        (True, False, -1, 5, "start and duration are invalid"),
        (True, False, 0, 0, "start and duration are invalid"),
        (True, False, 16, 5, "source has ended"),
    ],
)
def test_invalid_clip_requests_are_rejected(
    tmp_path: Path,
    source_exists: bool,
    same_destination: bool,
    start_s: float,
    duration_s: float,
    message: str,
) -> None:
    source = tmp_path / "source.mp4"
    if source_exists:
        source.write_bytes(b"original-video")
    destination = source if same_destination else tmp_path / "capture.mp4"

    with pytest.raises(InvalidComponentData, match=message):
        asyncio.run(
            build_clipper(FakeMediaCommands()).extract(
                source,
                destination,
                start_s=start_s,
                duration_s=duration_s,
            )
        )


@pytest.mark.parametrize(
    ("commands", "message"),
    [
        (FakeMediaCommands(fail_probe=True), "cannot probe media: probe failure"),
        (FakeMediaCommands(invalid_probe_json=True), "invalid media metadata"),
        (FakeMediaCommands(output_duration_s=4.5), "duration differs from request"),
    ],
)
def test_probe_and_duration_validation_fail_closed(
    tmp_path: Path,
    commands: FakeMediaCommands,
    message: str,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original-video")

    with pytest.raises(InvalidComponentData, match=message):
        asyncio.run(
            build_clipper(commands).extract(
                source,
                tmp_path / "capture.mp4",
                start_s=1,
                duration_s=5,
            )
        )

    assert not list(tmp_path.glob("*.partial.mp4"))


def test_missing_media_executable_is_a_stable_component_failure() -> None:
    with pytest.raises(ComponentUnavailable, match="media executable is unavailable"):
        FfmpegVideoClipper._run_command(["definitely-missing-bearvision-tool"])


def test_negative_duration_tolerance_is_rejected() -> None:
    with pytest.raises(ValueError, match="duration tolerance"):
        FfmpegVideoClipper(ClipExtractionConfig(), duration_tolerance_s=-0.1)


def test_executable_resolution_prefers_packaged_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BEARVISION_TEST_MEDIA", raising=False)
    monkeypatch.setattr("bearvision.adapters.ffmpeg.shutil.which", lambda _: "managed-tool.exe")

    resolved = FfmpegVideoClipper._resolve_executable(
        None,
        "BEARVISION_TEST_MEDIA",
        "ffmpeg",
    )

    assert Path(resolved).stem == "ffmpeg"
    assert resolved != "managed-tool.exe"


def test_executable_resolution_falls_back_to_system_path_without_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = __import__

    def import_without_media_package(name, *args, **kwargs):
        if name == "ffmpeg_binaries":
            raise ImportError("simulated missing optional package")
        return real_import(name, *args, **kwargs)

    monkeypatch.delenv("BEARVISION_TEST_MEDIA", raising=False)
    monkeypatch.setattr("builtins.__import__", import_without_media_package)
    monkeypatch.setattr("bearvision.adapters.ffmpeg.shutil.which", lambda _: "managed-tool.exe")

    resolved = FfmpegVideoClipper._resolve_executable(
        None,
        "BEARVISION_TEST_MEDIA",
        "ffprobe",
    )

    assert resolved == "managed-tool.exe"
