import asyncio
from datetime import datetime, timezone

from bearvision.contracts import (
    BoundingBox,
    CaptureRequest,
    PersonDetection,
    TagObservation,
    Vector3,
)
from bearvision.ports import VideoFrame
from bearvision.simulation import (
    InMemoryStorage,
    SimulatedCamera,
    SimulatedDetector,
    SimulatedTagScanner,
    VirtualClock,
)
from bearvision.testing import (
    check_camera,
    check_clock,
    check_detector,
    check_scanner,
    check_storage,
)


NOW = datetime(2026, 8, 12, tzinfo=timezone.utc)


def test_simulated_adapters_pass_shared_contract_suites() -> None:
    clock = VirtualClock(NOW)
    request = CaptureRequest(
        request_id="capture-1",
        requested_at_monotonic_s=1,
        pre_roll_s=15,
        post_roll_s=5,
    )
    observation = TagObservation(
        tag_id="tag-17",
        observed_at_utc=NOW,
        observed_at_monotonic_s=1,
        rssi_dbm=-50,
        acceleration_mps2=Vector3(x=0, y=0, z=9.81),
    )
    frame = VideoFrame("frame-1", 1, 100, 100, b"pixels")
    detection = PersonDetection(
        frame_id="frame-1",
        observed_at_monotonic_s=1,
        bounding_box=BoundingBox(x_px=1, y_px=1, width_px=20, height_px=30),
        confidence=0.9,
    )
    camera = SimulatedCamera(clock)
    storage = InMemoryStorage(clock)

    asyncio.run(check_clock(clock))
    capture = asyncio.run(check_camera(camera, request))
    asyncio.run(check_scanner(SimulatedTagScanner((observation,))))
    asyncio.run(check_detector(SimulatedDetector({"frame-1": (detection,)}), frame))
    asyncio.run(check_storage(storage, capture.media, "rider-17/clip.mp4"))
