import asyncio
from pathlib import Path

import pytest

from bearvision.adapters import GoProCameraAdapter
from bearvision.contracts import CaptureRequest
from bearvision.ports import ExtractedClip, InvalidComponentData
from bearvision.simulation import SimulatedGoProController, VirtualClock


class RecordingClipper:
    def __init__(self) -> None:
        self.requests: list[tuple[Path, Path, float, float]] = []

    async def extract(
        self,
        source: Path,
        destination: Path,
        *,
        start_s: float,
        duration_s: float,
    ) -> ExtractedClip:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"simulated-mp4:start={start_s}:duration={duration_s}".encode())
        self.requests.append((source, destination, start_s, duration_s))
        return ExtractedClip(
            path=destination,
            start_s=start_s,
            duration_s=duration_s,
            width_px=1920,
            height_px=1080,
            has_audio=True,
        )


def build_controller(
    tmp_path: Path,
    clock: VirtualClock,
    clipper: RecordingClipper,
) -> SimulatedGoProController:
    preview = tmp_path / "preview.mp4"
    preview.write_bytes(b"source-video")
    return SimulatedGoProController(
        root_dir=tmp_path / "sd-card",
        preview_source=preview,
        clock=clock,
        clipper=clipper,
    )


def test_simulated_gopro_models_preview_hindsight_media_and_persistence(
    tmp_path: Path,
) -> None:
    clock = VirtualClock()
    clipper = RecordingClipper()
    controller = build_controller(tmp_path, clock, clipper)

    with pytest.raises(ConnectionError, match="disconnected"):
        controller.list_videos()

    controller.connect()
    assert controller.enable_hindsight(15)
    clock.advance_to(20)
    assert controller.start_preview() == str((tmp_path / "preview.mp4").resolve())
    assert controller.get_camera_status()["status"]["preview_stream"] is True
    controller.stop_preview()

    controller.start_recording()
    clock.advance_by(5)
    controller.stop_recording()

    assert [(start, duration) for _, _, start, duration in clipper.requests] == [(5, 20)]
    assert controller.list_videos() == ["100GOPRO/GX010001.MP4"]
    downloaded = controller.download_file("100GOPRO/GX010001.MP4", str(tmp_path / "downloaded.mp4"))
    assert downloaded.read_bytes().startswith(b"simulated-mp4")
    controller.disconnect()

    restarted = build_controller(tmp_path, clock, clipper)
    restarted.connect()
    assert restarted.get_camera_status()["settings"]["hindsight_duration_s"] == 15
    assert restarted.list_videos() == ["100GOPRO/GX010001.MP4"]
    restarted.disconnect()


def test_production_adapter_runs_against_disk_backed_gopro_simulator(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        clock = VirtualClock()
        clipper = RecordingClipper()
        controller = build_controller(tmp_path, clock, clipper)
        camera = GoProCameraAdapter(
            controller,
            clock,
            tmp_path / "downloads",
            hindsight_enabled=True,
            hindsight_duration_s=15,
        )
        request = CaptureRequest(
            request_id="capture-1",
            requested_at_monotonic_s=20,
            pre_roll_s=15,
            post_roll_s=5,
        )

        await camera.connect()
        assert await camera.start_preview() == str((tmp_path / "preview.mp4").resolve())
        clock.advance_to(20)
        first = await camera.capture(request)
        second = await camera.capture(request)
        assert first == second
        assert first.media.local_path is not None
        assert first.media.local_path.read_bytes().startswith(b"simulated-mp4")
        assert first.media.asset.filename == "capture-1-GX010001.MP4"
        assert first.requested_window.start_monotonic_s == 5
        assert first.requested_window.end_monotonic_s == 25
        assert first.actual_window.start_monotonic_s == 5
        assert first.actual_window.end_monotonic_s == 25
        assert first.actual_window.precision == "exact"
        assert len(clipper.requests) == 1
        assert clipper.requests[0][2:] == (5, 20)
        assert clock.monotonic() == 25
        await camera.stop_preview()
        await camera.disconnect()

    asyncio.run(exercise())


def test_gopro_capture_clamps_pre_roll_to_available_startup_media(tmp_path: Path) -> None:
    async def exercise() -> None:
        clock = VirtualClock()
        clipper = RecordingClipper()
        camera = GoProCameraAdapter(
            build_controller(tmp_path, clock, clipper),
            clock,
            tmp_path / "downloads",
            hindsight_enabled=True,
            hindsight_duration_s=15,
        )
        await camera.connect()
        clock.advance_to(5)
        capture = await camera.capture(
            CaptureRequest(
                request_id="startup-capture",
                requested_at_monotonic_s=5,
                pre_roll_s=15,
                post_roll_s=2,
            )
        )

        assert capture.requested_window.start_monotonic_s == 0
        assert capture.actual_window.start_monotonic_s == 0
        assert capture.actual_window.end_monotonic_s == 7
        assert clipper.requests[0][2:] == (0, 7)
        await camera.disconnect()

    asyncio.run(exercise())


def test_gopro_capture_rejects_request_that_disagrees_with_hindsight(tmp_path: Path) -> None:
    async def exercise() -> None:
        clock = VirtualClock()
        camera = GoProCameraAdapter(
            build_controller(tmp_path, clock, RecordingClipper()),
            clock,
            tmp_path / "downloads",
            hindsight_enabled=True,
            hindsight_duration_s=15,
        )
        await camera.connect()
        with pytest.raises(
            InvalidComponentData,
            match="does not match configured GoPro HindSight",
        ):
            await camera.capture(
                CaptureRequest(
                    request_id="mismatch",
                    requested_at_monotonic_s=20,
                    pre_roll_s=30,
                    post_roll_s=2,
                )
            )
        await camera.disconnect()

    asyncio.run(exercise())


@pytest.mark.parametrize("hindsight_duration_s", [15, 30])
def test_gopro_capture_honours_configured_hindsight_window(
    tmp_path: Path,
    hindsight_duration_s: int,
) -> None:
    async def exercise() -> None:
        clock = VirtualClock()
        camera = GoProCameraAdapter(
            build_controller(tmp_path, clock, RecordingClipper()),
            clock,
            tmp_path / "downloads",
            hindsight_enabled=True,
            hindsight_duration_s=hindsight_duration_s,
        )
        await camera.connect()
        clock.advance_to(40)
        capture = await camera.capture(
            CaptureRequest(
                request_id=f"hindsight-{hindsight_duration_s}",
                requested_at_monotonic_s=40,
                pre_roll_s=hindsight_duration_s,
                post_roll_s=2,
            )
        )

        assert capture.requested_window.start_monotonic_s == 40 - hindsight_duration_s
        assert capture.actual_window.start_monotonic_s == 40 - hindsight_duration_s
        assert capture.actual_window.end_monotonic_s == 42
        await camera.disconnect()

    asyncio.run(exercise())


def test_simulated_gopro_rejects_invalid_hindsight_and_media_escape(
    tmp_path: Path,
) -> None:
    controller = build_controller(tmp_path, VirtualClock(), RecordingClipper())
    controller.connect()
    with pytest.raises(ValueError, match="15 or 30"):
        controller.enable_hindsight(5)
    with pytest.raises(ValueError, match="escapes"):
        controller.download_file("../outside.mp4", str(tmp_path / "copy.mp4"))
    controller.disconnect()
