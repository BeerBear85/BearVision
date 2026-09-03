import asyncio
from pathlib import Path
import threading
from types import SimpleNamespace

from open_gopro.models import proto
from open_gopro.models.constants import SettingId, StatusId, settings

from bearvision.adapters import (
    BoxJobQueue,
    BoxStorageAdapter,
    GoProCameraAdapter,
    KBeaconTagScannerAdapter,
    YoloDetectorAdapter,
)
from bearvision.config import load_edge_config
from bearvision.edge import BearVisionOrchestrator, build_real_orchestrator, build_real_system
from bearvision.contracts import CaptureRequest
from bearvision.integrations.async_gopro import AsyncGoProController
from bearvision.ports import VideoFrame
from bearvision.simulation import VirtualClock
from bearvision.processing import VirtualCameramanProcessor
from bearvision.testing import check_camera, check_detector, check_scanner, check_storage
from tests.stubs.gopro import FakeGoPro


class StubGoPro:
    def __init__(self) -> None:
        self.files: list[str] = []
        self.hindsight_duration_s = 0

    def connect(self):
        return None

    def disconnect(self):
        return None

    def start_preview(self):
        return "udp://camera:8554"

    def stop_preview(self):
        return None

    def list_videos(self):
        return list(self.files)

    def enable_hindsight(self, duration):
        self.hindsight_duration_s = duration
        return True

    def disableHindsightMode(self):
        self.hindsight_duration_s = 0

    def start_recording(self):
        return None

    def stop_recording(self):
        self.files.append("GX010001.MP4")

    def download_file(self, camera_file, local_path):
        path = Path(local_path)
        path.write_bytes(b"gopro-video")
        return path


class AsyncStubGoPro(StubGoPro):
    def __init__(self) -> None:
        super().__init__()
        self.thread_ids: list[int] = []

    async def connect(self):
        self.thread_ids.append(threading.get_ident())

    async def disconnect(self):
        self.thread_ids.append(threading.get_ident())

    async def start_preview(self):
        self.thread_ids.append(threading.get_ident())
        return "udp://camera:8554"

    async def stop_preview(self):
        self.thread_ids.append(threading.get_ident())

    async def enable_hindsight(self, duration):
        self.thread_ids.append(threading.get_ident())
        self.hindsight_duration_s = duration
        return True

    async def list_videos(self):
        self.thread_ids.append(threading.get_ident())
        return list(self.files)

    async def start_recording(self):
        self.thread_ids.append(threading.get_ident())

    async def stop_recording(self):
        self.thread_ids.append(threading.get_ident())
        self.files.append("GX010001.MP4")

    async def download_file(self, camera_file, local_path):
        self.thread_ids.append(threading.get_ident())
        path = Path(local_path)
        path.write_bytes(b"gopro-video")
        return path


class StubDnn:
    def __init__(self, model=None) -> None:
        self.model = model
        self.confidence_threshold = 0.0

    def init(self):
        return None

    def find_person(self, payload):
        return [[[1, 2, 30, 40]], [0.91]]


class StubBox:
    def __init__(self, config=None) -> None:
        self.config = config
        self.objects: dict[str, bytes] = {}

    def upload_file(self, local_path, remote_path, overwrite=False):
        if remote_path in self.objects and not overwrite:
            raise FileExistsError(remote_path)
        self.objects[remote_path] = Path(local_path).read_bytes()

    def download_file(self, remote_path, local_path):
        Path(local_path).write_bytes(self.objects[remote_path])

    def delete_file(self, remote_path):
        self.objects.pop(remote_path, None)


class StubBeacon:
    def __init__(self) -> None:
        self.advertisement_queue = asyncio.Queue()


def capture_request() -> CaptureRequest:
    return CaptureRequest(
        request_id="capture-1",
        requested_at_monotonic_s=1,
        pre_roll_s=15,
        post_roll_s=1,
    )


def test_existing_camera_detector_and_storage_wrappers_pass_contracts(tmp_path: Path) -> None:
    clock = VirtualClock()
    camera = GoProCameraAdapter(StubGoPro(), clock, tmp_path / "captures")
    capture = asyncio.run(check_camera(camera, capture_request()))
    asyncio.run(
        check_detector(
            YoloDetectorAdapter(StubDnn()),
            VideoFrame("frame-1", 1, 100, 100, b"pixels"),
        )
    )
    asyncio.run(
        check_storage(
            BoxStorageAdapter(StubBox(), clock, tmp_path / "scratch"),
            capture.media,
            "rider-17/clip.mp4",
        )
    )


def test_async_camera_integration_stays_on_the_runtime_event_loop(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime_thread = threading.get_ident()
        controller = AsyncStubGoPro()
        camera = GoProCameraAdapter(controller, VirtualClock(), tmp_path / "captures")

        await check_camera(camera, capture_request())

        assert controller.thread_ids
        assert set(controller.thread_ids) == {runtime_thread}

    asyncio.run(exercise())


def test_async_gopro_controller_maps_the_sdk_without_a_thread_bridge(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        sdk = FakeGoPro()
        controller = AsyncGoProController(gopro=sdk)
        destination = tmp_path / "download.mp4"

        await controller.connect()
        assert await controller.enable_hindsight(15)
        assert await controller.start_preview() == (
            "udp://@0.0.0.0:8554?fifo_size=65536&overrun_nonfatal=1"
        )
        await controller.stop_preview()
        await controller.start_recording()
        await controller.stop_recording()
        assert await controller.list_videos() == ["DCIM/100GOPRO/GOPR0001.MP4"]
        assert await controller.download_file("GOPR0001.MP4", str(destination)) == destination
        await controller.disconnect()

        assert destination.read_text(encoding="utf-8") == "data"

    asyncio.run(exercise())


def test_async_gopro_switches_to_video_before_camera_configuration() -> None:
    async def exercise() -> None:
        sdk = FakeGoPro()
        calls: list[object] = []

        async def open_camera():
            calls.append("open")

        async def load_preset_group(*, group):
            calls.append(group)
            return SimpleNamespace(ok=True)

        async def camera_state():
            return SimpleNamespace(
                ok=True,
                data={StatusId.PRESET_GROUP: proto.EnumPresetGroup.PRESET_GROUP_ID_PHOTO},
            )

        sdk.open = open_camera
        sdk.http_command.get_camera_state = camera_state
        sdk.http_command.load_preset_group = load_preset_group

        await AsyncGoProController(gopro=sdk).connect()

        assert calls == ["open", proto.EnumPresetGroup.PRESET_GROUP_ID_VIDEO]

    asyncio.run(exercise())


def test_async_gopro_does_not_reapply_matching_video_preset_group() -> None:
    async def exercise() -> None:
        sdk = FakeGoPro()
        writes = 0

        async def camera_state():
            return SimpleNamespace(
                ok=True,
                data={StatusId.PRESET_GROUP: proto.EnumPresetGroup.PRESET_GROUP_ID_VIDEO},
            )

        async def reject_redundant_write(*, group):
            nonlocal writes
            writes += 1
            return SimpleNamespace(ok=False)

        sdk.http_command.get_camera_state = camera_state
        sdk.http_command.load_preset_group = reject_redundant_write

        await AsyncGoProController(gopro=sdk).connect()

        assert writes == 0

    asyncio.run(exercise())


def test_async_gopro_does_not_reapply_matching_hindsight_setting() -> None:
    async def exercise() -> None:
        sdk = FakeGoPro()
        writes = 0

        async def camera_state():
            return SimpleNamespace(
                ok=True,
                data={SettingId.HINDSIGHT: settings.Hindsight.NUM_15_SECONDS},
            )

        async def reject_redundant_write(value):
            nonlocal writes
            writes += 1
            return SimpleNamespace(ok=False)

        sdk.http_command.get_camera_state = camera_state
        sdk.http_setting.hindsight.set = reject_redundant_write
        controller = AsyncGoProController(gopro=sdk)

        assert await controller.enable_hindsight(15)
        assert writes == 0

    asyncio.run(exercise())


def test_async_gopro_verifies_hindsight_after_rejected_write() -> None:
    async def exercise() -> None:
        sdk = FakeGoPro()
        states = [
            settings.Hindsight.OFF,
            settings.Hindsight.NUM_15_SECONDS,
        ]

        async def camera_state():
            return SimpleNamespace(
                ok=True,
                data={SettingId.HINDSIGHT: states.pop(0)},
            )

        async def rejected_write(value):
            return SimpleNamespace(ok=False)

        sdk.http_command.get_camera_state = camera_state
        sdk.http_setting.hindsight.set = rejected_write
        controller = AsyncGoProController(gopro=sdk)

        assert await controller.enable_hindsight(15)
        assert states == []

    asyncio.run(exercise())


def test_existing_beacon_wrapper_converts_acceleration_and_voltage() -> None:
    async def exercise() -> None:
        handler = StubBeacon()
        await handler.advertisement_queue.put(
            {
                "address": "tag-17",
                "rssi": -52,
                "batteryLevel": 3000,
                "acc_sensor": SimpleNamespace(x=0.0, y=0.0, z=1.0),
            }
        )
        scanner = KBeaconTagScannerAdapter(
            handler,
            VirtualClock(),
            manage_scan=False,
            maximum_observations=1,
        )
        await check_scanner(scanner)
        await handler.advertisement_queue.join()

    asyncio.run(exercise())


def test_real_composition_wraps_existing_implementations(tmp_path: Path) -> None:
    config = load_edge_config(Path(__file__).resolve().parents[2] / "config" / "edge.yaml")

    components = build_real_system(
        config,
        capture_dir=tmp_path / "captures",
        scratch_dir=tmp_path / "scratch",
        gopro_factory=StubGoPro,
        beacon_factory=StubBeacon,
        detector_factory=StubDnn,
        box_factory=StubBox,
    )

    assert isinstance(components.camera, GoProCameraAdapter)
    assert isinstance(components.scanner, KBeaconTagScannerAdapter)
    assert isinstance(components.detector, YoloDetectorAdapter)
    assert isinstance(components.job_queue, BoxJobQueue)
    assert isinstance(components.clip_processor, VirtualCameramanProcessor)
    assert components.clip_processor.config == config.virtual_cameraman
    assert components.detector.handler is not components.clip_processor.detector.handler
    assert (
        components.detector.handler.confidence_threshold
        == components.clip_processor.detector.handler.confidence_threshold
        == config.detection.confidence_threshold
    )

    orchestrator = build_real_orchestrator(
        config,
        capture_dir=tmp_path / "captures",
        scratch_dir=tmp_path / "scratch",
        gopro_factory=StubGoPro,
        beacon_factory=StubBeacon,
        detector_factory=StubDnn,
        box_factory=StubBox,
    )
    assert isinstance(orchestrator, BearVisionOrchestrator)
    assert orchestrator.recording_duration_s == config.recording.post_detection_duration_s
    assert orchestrator.capture_pre_roll_s == config.recording.hindsight_duration_s
    assert orchestrator.raw_clip_pipeline is not None
