import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from bearvision.edge.hardware_readiness import PhysicalReadinessHandshake
from bearvision.edge.preflight import ProbeOutcome, check_edge_readiness
from bearvision.ports import VideoFrame


ROOT = Path(__file__).resolve().parents[2]


def test_physical_handshake_receives_preview_frame_and_runs_ble_scan() -> None:
    calls: list[str] = []

    class Camera:
        async def connect(self) -> None:
            calls.append("camera.connect")

        async def start_preview(self) -> str:
            calls.append("camera.start_preview")
            return "udp://camera:8554"

        async def stop_preview(self) -> None:
            calls.append("camera.stop_preview")

        async def disconnect(self) -> None:
            calls.append("camera.disconnect")

    class Frames:
        async def open(self, source: str) -> None:
            calls.append(f"frames.open:{source}")

        async def frames(self):
            yield VideoFrame("preview-1", 1.0, 1920, 1080, b"pixels")

        async def close(self) -> None:
            calls.append("frames.close")

    class BleSource:
        def __init__(self) -> None:
            self.advertisement_queue = SimpleNamespace(qsize=lambda: 2)

        async def look_for_advertisements(
            self, timeout: float, *, stop_timeout: float | None = None
        ) -> None:
            calls.append(f"ble.scan:{timeout}:cleanup={stop_timeout}")

    handshake = PhysicalReadinessHandshake(
        camera_factory=Camera,
        frame_source_factory=Frames,
        ble_source_factory=BleSource,
        camera_timeout_s=1,
        ble_scan_duration_s=0.25,
        cleanup_timeout_s=1,
    )

    async def exercise() -> tuple[str, str]:
        return await handshake.check_camera_preview(), await handshake.check_ble_scanner()

    camera_evidence, ble_evidence = asyncio.run(exercise())

    assert camera_evidence == "GoPro preview received a 1920x1080 frame"
    assert ble_evidence == "BLE scanner completed; 2 BearTag advertisements observed"
    assert calls == [
        "camera.connect",
        "camera.start_preview",
        "frames.open:udp://camera:8554",
        "frames.close",
        "camera.stop_preview",
        "camera.disconnect",
        "ble.scan:0.25:cleanup=1",
    ]


def test_preview_timeout_still_closes_stream_and_disconnects_camera() -> None:
    calls: list[str] = []

    class Camera:
        async def connect(self) -> None:
            calls.append("connect")

        async def start_preview(self) -> str:
            calls.append("start_preview")
            return "udp://camera:8554"

        async def stop_preview(self) -> None:
            calls.append("stop_preview")

        async def disconnect(self) -> None:
            calls.append("disconnect")

    class HangingFrames:
        async def open(self, _source: str) -> None:
            calls.append("open")

        async def frames(self):
            await asyncio.Future()
            yield  # pragma: no cover

        async def close(self) -> None:
            calls.append("close")

    handshake = PhysicalReadinessHandshake(
        camera_factory=Camera,
        frame_source_factory=HangingFrames,
        ble_source_factory=lambda: None,
        camera_timeout_s=0.01,
        ble_scan_duration_s=0.01,
        cleanup_timeout_s=1,
    )

    with pytest.raises(TimeoutError, match="GoPro preview handshake did not complete within 0.01 seconds"):
        asyncio.run(handshake.check_camera_preview())

    assert calls == ["connect", "start_preview", "open", "close", "stop_preview", "disconnect"]


def test_ble_timeout_has_operator_facing_evidence() -> None:
    class HangingBleSource:
        advertisement_queue = SimpleNamespace(qsize=lambda: 0)

        async def look_for_advertisements(
            self, timeout: float, *, stop_timeout: float | None = None
        ) -> None:
            await asyncio.Future()

    handshake = PhysicalReadinessHandshake(
        camera_factory=lambda: None,
        frame_source_factory=lambda: None,
        ble_source_factory=HangingBleSource,
        camera_timeout_s=1,
        ble_scan_duration_s=0.01,
        cleanup_timeout_s=0.01,
    )

    with pytest.raises(TimeoutError, match="BLE scanner handshake did not complete within"):
        asyncio.run(handshake.check_ble_scanner())


def test_ble_handshake_passes_when_no_bear_tags_are_observed() -> None:
    class EmptyBleSource:
        advertisement_queue = SimpleNamespace(qsize=lambda: 0)

        async def look_for_advertisements(
            self, timeout: float, *, stop_timeout: float | None = None
        ) -> None:
            pass

    handshake = PhysicalReadinessHandshake(
        camera_factory=lambda: None,
        frame_source_factory=lambda: None,
        ble_source_factory=EmptyBleSource,
        camera_timeout_s=1,
        ble_scan_duration_s=0.01,
        cleanup_timeout_s=0.01,
    )

    evidence = asyncio.run(handshake.check_ble_scanner())

    assert evidence == "BLE scanner completed; 0 BearTag advertisements observed"


def test_failed_camera_connection_still_attempts_disconnect() -> None:
    calls: list[str] = []

    class FailingCamera:
        async def connect(self) -> None:
            calls.append("connect")
            raise ConnectionError("camera unavailable")

        async def disconnect(self) -> None:
            calls.append("disconnect")

    handshake = PhysicalReadinessHandshake(
        camera_factory=FailingCamera,
        frame_source_factory=lambda: None,
        ble_source_factory=lambda: None,
        camera_timeout_s=1,
        ble_scan_duration_s=1,
        cleanup_timeout_s=1,
    )

    with pytest.raises(ConnectionError, match="camera unavailable"):
        asyncio.run(handshake.check_camera_preview())

    assert calls == ["connect", "disconnect"]


def test_cleanup_timeout_is_reported_after_successful_preview() -> None:
    class Camera:
        async def connect(self) -> None:
            pass

        async def start_preview(self) -> str:
            return "udp://camera:8554"

        async def stop_preview(self) -> None:
            pass

        async def disconnect(self) -> None:
            await asyncio.Future()

    class Frames:
        async def open(self, _source: str) -> None:
            pass

        async def frames(self):
            yield VideoFrame("preview-1", 1.0, 640, 480, b"pixels")

        async def close(self) -> None:
            pass

    handshake = PhysicalReadinessHandshake(
        camera_factory=Camera,
        frame_source_factory=Frames,
        ble_source_factory=lambda: None,
        camera_timeout_s=1,
        ble_scan_duration_s=1,
        cleanup_timeout_s=0.01,
    )

    with pytest.raises(TimeoutError, match="disconnecting camera"):
        asyncio.run(handshake.check_camera_preview())


def outcomes(**statuses: str):
    return {
        check_id: (lambda status=status: ProbeOutcome(status, f"{check_id} evidence"))
        for check_id, status in statuses.items()
    }


def test_critical_readiness_failure_blocks_start_with_corrective_action(tmp_path: Path) -> None:
    report = check_edge_readiness(
        ROOT / "config" / "edge.yaml",
        capture_dir=tmp_path / "captures",
        scratch_dir=tmp_path / "scratch",
        probe_overrides=outcomes(
            runtime="pass",
            model="pass",
            media_tools="pass",
            capture_storage="pass",
            scratch_storage="pass",
            camera="fail",
            ble="pass",
            cloud_storage="pass",
        ),
    )

    camera = next(check for check in report.checks if check.check_id == "camera")
    assert report.blocking
    assert camera.critical
    assert camera.status == "fail"
    assert camera.corrective_action


def test_readiness_warning_is_overridable_and_does_not_block_start(tmp_path: Path) -> None:
    report = check_edge_readiness(
        ROOT / "config" / "edge.yaml",
        capture_dir=tmp_path / "captures",
        scratch_dir=tmp_path / "scratch",
        probe_overrides=outcomes(
            runtime="pass",
            model="pass",
            media_tools="pass",
            capture_storage="pass",
            scratch_storage="pass",
            camera="pass",
            ble="warning",
            cloud_storage="pass",
        ),
    )

    assert not report.blocking
    assert report.warning_ids == ("ble",)
    assert report.readiness_schema_version == "1.0"


def test_readiness_uses_real_handshake_evidence_for_camera_and_ble(tmp_path: Path) -> None:
    calls: list[str] = []

    class Handshake:
        async def check_camera_preview(self) -> str:
            calls.append("camera")
            return "GoPro preview received a 1280x720 frame"

        async def check_ble_scanner(self) -> str:
            calls.append("ble")
            return "BLE scanner completed; 0 BearTag advertisements observed"

    report = check_edge_readiness(
        ROOT / "config" / "edge.yaml",
        capture_dir=tmp_path / "captures",
        scratch_dir=tmp_path / "scratch",
        hardware_handshake=Handshake(),
        probe_overrides=outcomes(
            runtime="pass",
            model="pass",
            media_tools="pass",
            capture_storage="pass",
            scratch_storage="pass",
            cloud_storage="pass",
        ),
    )

    camera = next(check for check in report.checks if check.check_id == "camera")
    ble = next(check for check in report.checks if check.check_id == "ble")
    assert not report.blocking
    assert camera.status == "pass"
    assert camera.evidence == "GoPro preview received a 1280x720 frame"
    assert ble.status == "pass"
    assert calls == ["camera", "ble"]


def test_handshake_failures_are_independent_blocking_checks(tmp_path: Path) -> None:
    class FailingHandshake:
        async def check_camera_preview(self) -> str:
            raise TimeoutError("GoPro preview produced no frame before the 12 second timeout")

        async def check_ble_scanner(self) -> str:
            raise RuntimeError("BLE scanner could not start")

    report = check_edge_readiness(
        ROOT / "config" / "edge.yaml",
        capture_dir=tmp_path / "captures",
        scratch_dir=tmp_path / "scratch",
        hardware_handshake=FailingHandshake(),
        probe_overrides=outcomes(
            runtime="pass",
            model="pass",
            media_tools="pass",
            capture_storage="pass",
            scratch_storage="pass",
            cloud_storage="pass",
        ),
    )

    checks = {check.check_id: check for check in report.checks}
    assert report.blocking
    assert checks["camera"].status == "fail"
    assert "no frame" in checks["camera"].evidence
    assert checks["ble"].status == "fail"
    assert checks["camera"].corrective_action
    assert checks["ble"].corrective_action
