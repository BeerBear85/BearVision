import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from bearvision.adapters import (
    BleakKBeaconSource,
    BoxStorageAdapter,
    GoProCameraAdapter,
    KBeaconTagScannerAdapter,
)
from bearvision.adapters._errors import translated_error
from bearvision.contracts import CaptureRequest, MediaAsset
from bearvision.ports import (
    CapturedMedia,
    ComponentError,
    ComponentTimeout,
    ComponentUnavailable,
    InvalidComponentData,
)
from bearvision.simulation import VirtualClock


def request(request_id: str = "capture-1") -> CaptureRequest:
    return CaptureRequest(
        request_id=request_id,
        requested_at_monotonic_s=1,
        pre_roll_s=0,
        post_roll_s=1,
    )


class RecordingGoPro:
    def __init__(self, *, download_exists: bool = True) -> None:
        self.files: list[str] = []
        self.download_exists = download_exists
        self.started = 0
        self.stopped = 0

    def list_videos(self):
        return list(self.files)

    def start_recording(self):
        self.started += 1
        self.files.append("GX010099.MP4")

    def stop_recording(self):
        self.stopped += 1

    def download_file(self, camera_file, local_path):
        path = Path(local_path)
        if self.download_exists:
            path.write_bytes(b"recorded")
        return path


class StaticMediaProbe:
    def __init__(self, duration_s: float) -> None:
        self.duration_s = duration_s

    async def duration(self, source: Path) -> float:
        assert source.is_file()
        return self.duration_s


def test_gopro_non_hindsight_capture_records_waits_and_stops(tmp_path: Path) -> None:
    controller = RecordingGoPro()
    clock = VirtualClock()
    camera = GoProCameraAdapter(
        controller,
        clock,
        tmp_path,
        hindsight_enabled=False,
    )

    capture = asyncio.run(camera.capture(request()))

    assert controller.started == 1
    assert controller.stopped == 1
    assert clock.monotonic() == 1
    assert (
        capture.media.local_path is not None
        and capture.media.local_path.read_bytes() == b"recorded"
    )


def test_gopro_capture_rejects_missing_new_file_and_missing_download(tmp_path: Path) -> None:
    no_media = RecordingGoPro()
    no_media.start_recording = lambda: None
    camera = GoProCameraAdapter(no_media, VirtualClock(), tmp_path, hindsight_enabled=False)
    with pytest.raises(InvalidComponentData, match="produced no new media"):
        asyncio.run(camera.capture(request("no-media")))

    missing_download = RecordingGoPro(download_exists=False)
    camera = GoProCameraAdapter(
        missing_download,
        VirtualClock(),
        tmp_path,
        hindsight_enabled=False,
    )
    with pytest.raises(InvalidComponentData, match="download is missing"):
        asyncio.run(camera.capture(request("missing-download")))


def test_physical_gopro_timing_is_estimated_from_commands_and_media_duration(
    tmp_path: Path,
) -> None:
    clock = VirtualClock()
    clock.advance_to(10)
    camera = GoProCameraAdapter(
        RecordingGoPro(),
        clock,
        tmp_path,
        hindsight_enabled=False,
        media_probe=StaticMediaProbe(2),
    )
    capture = asyncio.run(
        camera.capture(
            CaptureRequest(
                request_id="capture-probed",
                requested_at_monotonic_s=10,
                pre_roll_s=0,
                post_roll_s=2,
            )
        )
    )

    assert capture.requested_window.start_monotonic_s == 10
    assert capture.requested_window.end_monotonic_s == 12
    assert capture.actual_window.start_monotonic_s == 10
    assert capture.actual_window.end_monotonic_s == 12
    assert capture.actual_window.precision == "estimated"
    assert capture.actual_window.basis == "camera_command_timing_and_media_duration"


def test_physical_gopro_rejects_media_duration_outside_command_bounds(
    tmp_path: Path,
) -> None:
    camera = GoProCameraAdapter(
        RecordingGoPro(),
        VirtualClock(),
        tmp_path,
        hindsight_enabled=False,
        media_probe=StaticMediaProbe(5),
    )

    with pytest.raises(InvalidComponentData, match="outside the camera-command bounds"):
        asyncio.run(camera.capture(request("wrong-duration")))


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        ("connect", "connect GoPro"),
        ("disconnect", "disconnect GoPro"),
        ("start_preview", "start GoPro preview"),
        ("stop_preview", "stop GoPro preview"),
    ],
)
def test_gopro_lifecycle_failures_are_typed(
    tmp_path: Path,
    operation: str,
    message: str,
) -> None:
    class FailingLifecycleGoPro:
        def connect(self):
            raise ConnectionError("offline")

        def disconnect(self):
            raise ConnectionError("offline")

        def start_preview(self):
            raise ConnectionError("offline")

        def stop_preview(self):
            raise ConnectionError("offline")

    camera = GoProCameraAdapter(FailingLifecycleGoPro(), VirtualClock(), tmp_path)

    with pytest.raises(ComponentUnavailable, match=message):
        asyncio.run(getattr(camera, operation)())


class MemoryBox:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.upload_calls = 0

    def upload_file(self, local_path, remote_path, overwrite=False):
        self.upload_calls += 1
        if remote_path in self.objects and not overwrite:
            raise FileExistsError(remote_path)
        self.objects[remote_path] = Path(local_path).read_bytes()

    def download_file(self, remote_path, local_path):
        Path(local_path).write_bytes(self.objects[remote_path])

    def delete_file(self, remote_path):
        self.objects.pop(remote_path, None)


def content_media(asset_id: str = "asset-1") -> CapturedMedia:
    return CapturedMedia(
        asset=MediaAsset(
            asset_id=asset_id,
            filename="clip.mp4",
            content_type="video/mp4",
            size_bytes=4,
            created_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        content=b"clip",
    )


def test_box_content_upload_cleans_scratch_and_delete_invalidates_cache(tmp_path: Path) -> None:
    handler = MemoryBox()
    storage = BoxStorageAdapter(handler, VirtualClock(), tmp_path)
    media = content_media()

    first = asyncio.run(storage.upload(media, "rider/clip.mp4"))
    second = asyncio.run(storage.upload(media, "rider/clip.mp4"))
    assert first == second
    assert handler.upload_calls == 1
    assert not list(tmp_path.iterdir())

    asyncio.run(storage.delete("rider/clip.mp4"))
    asyncio.run(storage.upload(media, "rider/clip.mp4"))
    assert handler.upload_calls == 2
    assert asyncio.run(storage.download("rider/clip.mp4")) == b"clip"
    assert not list(tmp_path.iterdir())


def test_box_overwrite_and_failures_are_translated(tmp_path: Path) -> None:
    handler = MemoryBox()
    storage = BoxStorageAdapter(handler, VirtualClock(), tmp_path)
    asyncio.run(storage.upload(content_media("asset-1"), "same.mp4"))
    overwritten = asyncio.run(storage.upload(content_media("asset-2"), "same.mp4", overwrite=True))
    assert overwritten.asset_id == "asset-2"

    class FailingBox(MemoryBox):
        def upload_file(self, local_path, remote_path, overwrite=False):
            raise ConnectionError("offline")

        def download_file(self, remote_path, local_path):
            raise TimeoutError("slow")

        def delete_file(self, remote_path):
            raise ValueError("bad key")

    failing = BoxStorageAdapter(FailingBox(), VirtualClock(), tmp_path)
    with pytest.raises(ComponentUnavailable, match="upload media to Box"):
        asyncio.run(failing.upload(content_media(), "clip.mp4"))
    with pytest.raises(ComponentTimeout, match="download media from Box"):
        asyncio.run(failing.download("clip.mp4"))
    with pytest.raises(InvalidComponentData, match="delete media from Box"):
        asyncio.run(failing.delete("clip.mp4"))
    assert not list(tmp_path.iterdir())


def ksensor_packet() -> bytes:
    return b"".join(
        [
            bytes([0x21]),
            (0x0009).to_bytes(2, "big"),
            (3000).to_bytes(2, "big"),
            b"\x00\x00",
            (1000).to_bytes(2, "big", signed=True),
            (-500).to_bytes(2, "big", signed=True),
            (250).to_bytes(2, "big", signed=True),
        ]
    )


def test_bleak_source_filters_and_decodes_kbeacon_advertisements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = []

    class FakeBleakScanner:
        def __init__(self, detection_callback) -> None:
            self.callback = detection_callback
            self.started = False
            self.stopped = False
            instances.append(self)

        async def start(self) -> None:
            self.started = True
            await self.callback(
                SimpleNamespace(name="other", address="ignored"),
                SimpleNamespace(service_data={"x": ksensor_packet()}, rssi=-80),
            )
            await self.callback(
                SimpleNamespace(name="KBPro", address="ignored-kbpro"),
                SimpleNamespace(service_data={"sensor": ksensor_packet()}, rssi=-40),
            )
            await self.callback(
                SimpleNamespace(name="bear_tag_17", address="physical-address"),
                SimpleNamespace(
                    service_data={"invalid": b"\x00", "sensor": ksensor_packet()},
                    rssi=-52,
                ),
            )

        async def stop(self) -> None:
            self.stopped = True

    bleak = ModuleType("bleak")
    bleak.BleakScanner = FakeBleakScanner
    monkeypatch.setitem(sys.modules, "bleak", bleak)
    source = BleakKBeaconSource()

    asyncio.run(source.look_for_advertisements(timeout=0.001))
    raw = source.advertisement_queue.get_nowait()

    assert raw["tag_id"] == "bear_tag_17"
    assert raw["address"] == "physical-address"
    assert raw["batteryLevel"] == 3000
    assert raw["acc_sensor"].x == 1.0
    assert raw["acc_sensor"].y == -0.5
    assert instances[0].started and instances[0].stopped

    async def cancel_continuous_scan() -> None:
        continuous = BleakKBeaconSource()
        task = asyncio.create_task(continuous.look_for_advertisements(timeout=0.0))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_continuous_scan())
    assert instances[-1].stopped is True


def test_kbeacon_managed_scan_is_cancelled_and_malformed_data_is_typed() -> None:
    async def exercise() -> None:
        class ManagedBeacon:
            def __init__(self) -> None:
                self.advertisement_queue = asyncio.Queue()
                self.started = asyncio.Event()
                self.cancelled = False

            async def look_for_advertisements(self, timeout=0.0) -> None:
                self.started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise

        managed = ManagedBeacon()
        scanner = KBeaconTagScannerAdapter(
            managed,
            VirtualClock(),
            manage_scan=True,
            maximum_observations=1,
        )
        stream = scanner.observations()
        observation_task = asyncio.create_task(anext(stream))
        await managed.started.wait()
        await managed.advertisement_queue.put(
            {
                "address": "tag-1",
                "rssi": -60,
                "acc_sensor": SimpleNamespace(x=0, y=0, z=1),
            }
        )
        observed = await observation_task
        await stream.aclose()
        assert observed.battery_voltage_mv is None
        assert managed.cancelled is True

        malformed = SimpleNamespace(advertisement_queue=asyncio.Queue())
        await malformed.advertisement_queue.put({"address": "tag-bad", "rssi": -60})
        scanner = KBeaconTagScannerAdapter(
            malformed,
            VirtualClock(),
            manage_scan=False,
            maximum_observations=1,
        )
        with pytest.raises(InvalidComponentData, match="decode KBeacon observation"):
            await anext(scanner.observations())
        await malformed.advertisement_queue.join()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (ComponentTimeout("typed"), ComponentTimeout),
        (TimeoutError("slow"), ComponentTimeout),
        (ConnectionError("offline"), ComponentUnavailable),
        (KeyError("field"), InvalidComponentData),
        (RuntimeError("unknown"), ComponentError),
    ],
)
def test_adapter_errors_have_stable_categories(exception: Exception, expected: type) -> None:
    translated = translated_error(exception, "operation")
    assert isinstance(translated, expected)
    if isinstance(exception, ComponentError):
        assert translated is exception
