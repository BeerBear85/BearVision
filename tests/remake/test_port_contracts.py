import asyncio
import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from bearvision.contracts import (
    BoundingBox,
    CaptureRequest,
    MediaAsset,
    PersonDetection,
    StorageReceipt,
    TagObservation,
    Vector3,
)
from bearvision.ports import (
    Camera,
    CapturedClip,
    CapturedMedia,
    CaptureWindow,
    CaptureWindowBasis,
    CaptureWindowPrecision,
    Clock,
    Detector,
    Storage,
    TagScanner,
    VideoFrame,
)
from bearvision.testing import (
    check_camera,
    check_clock,
    check_detector,
    check_scanner,
    check_storage,
)


NOW = datetime(2026, 8, 12, tzinfo=timezone.utc)


class ReferenceClock:
    def __init__(self) -> None:
        self.elapsed_s = 0.0

    def utc_now(self) -> datetime:
        return NOW + timedelta(seconds=self.elapsed_s)

    def monotonic(self) -> float:
        return self.elapsed_s

    async def sleep(self, delay_s: float) -> None:
        if delay_s < 0:
            raise ValueError("delay_s must not be negative")
        self.elapsed_s += delay_s


class ReferenceCamera:
    def __init__(self) -> None:
        self.captures: dict[str, CapturedClip] = {}

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def start_preview(self) -> str:
        return "memory://preview"

    async def stop_preview(self) -> None:
        return None

    async def capture(self, request: CaptureRequest) -> CapturedClip:
        if request.request_id not in self.captures:
            content = b"reference-video"
            requested_start_s = max(0.0, request.requested_at_monotonic_s - request.pre_roll_s)
            requested_end_s = request.requested_at_monotonic_s + request.post_roll_s
            self.captures[request.request_id] = CapturedClip(
                request_id=request.request_id,
                media=CapturedMedia(
                    asset=MediaAsset(
                        asset_id=f"asset-{request.request_id}",
                        filename=f"{request.request_id}.mp4",
                        content_type="video/mp4",
                        size_bytes=len(content),
                        created_at_utc=NOW,
                    ),
                    content=content,
                ),
                requested_window=CaptureWindow(
                    requested_start_s,
                    requested_end_s,
                    CaptureWindowPrecision.EXACT,
                    CaptureWindowBasis.DETECTION_REQUEST,
                ),
                actual_window=CaptureWindow(
                    requested_start_s,
                    requested_end_s,
                    CaptureWindowPrecision.EXACT,
                    CaptureWindowBasis.SIMULATED_MEDIA_TIMELINE,
                ),
            )
        return self.captures[request.request_id]


class ReferenceScanner:
    def __init__(self, items: tuple[TagObservation, ...]) -> None:
        self.items = items

    async def observations(self):
        for item in self.items:
            yield item


class ReferenceDetector:
    async def detect(self, frame: VideoFrame) -> tuple[PersonDetection, ...]:
        return (
            PersonDetection(
                frame_id=frame.frame_id,
                observed_at_monotonic_s=frame.observed_at_monotonic_s,
                bounding_box=BoundingBox(x_px=1, y_px=2, width_px=10, height_px=20),
                confidence=0.9,
            ),
        )


class ReferenceStorage:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, StorageReceipt]] = {}

    async def upload(
        self, media: CapturedMedia, object_key: str, *, overwrite: bool = False
    ) -> StorageReceipt:
        if object_key in self.objects:
            existing_content, receipt = self.objects[object_key]
            if receipt.asset_id == media.asset.asset_id:
                return receipt
            if not overwrite:
                raise FileExistsError(object_key)
        content = media.content or media.local_path.read_bytes()
        receipt = StorageReceipt(
            asset_id=media.asset.asset_id,
            object_key=object_key,
            stored_at_utc=NOW,
            checksum_sha256=hashlib.sha256(content).hexdigest(),
        )
        self.objects[object_key] = (content, receipt)
        return receipt

    async def download(self, object_key: str) -> bytes:
        return self.objects[object_key][0]

    async def delete(self, object_key: str) -> None:
        self.objects.pop(object_key, None)


def request() -> CaptureRequest:
    return CaptureRequest(
        request_id="capture-1",
        requested_at_monotonic_s=1,
        pre_roll_s=15,
        post_roll_s=5,
    )


def observation() -> TagObservation:
    return TagObservation(
        tag_id="tag-17",
        observed_at_utc=NOW,
        observed_at_monotonic_s=1,
        rssi_dbm=-52,
        acceleration_mps2=Vector3(x=0, y=0, z=9.81),
    )


def test_reference_components_satisfy_runtime_protocols() -> None:
    assert isinstance(ReferenceClock(), Clock)
    assert isinstance(ReferenceCamera(), Camera)
    assert isinstance(ReferenceScanner((observation(),)), TagScanner)
    assert isinstance(ReferenceDetector(), Detector)
    assert isinstance(ReferenceStorage(), Storage)


def test_reusable_component_contract_suites() -> None:
    clock = ReferenceClock()
    camera = ReferenceCamera()
    scanner = ReferenceScanner((observation(),))
    detector = ReferenceDetector()
    storage = ReferenceStorage()
    frame = VideoFrame("frame-1", 1, 1920, 1080, b"pixels")

    asyncio.run(check_clock(clock))
    capture = asyncio.run(check_camera(camera, request()))
    asyncio.run(check_scanner(scanner))
    asyncio.run(check_detector(detector, frame))
    asyncio.run(check_storage(storage, capture.media, "rider-17/capture-1.mp4"))


def test_captured_media_requires_exactly_one_source() -> None:
    asset = MediaAsset(
        asset_id="asset-1",
        filename="clip.mp4",
        content_type="video/mp4",
        size_bytes=0,
        created_at_utc=NOW,
    )
    with pytest.raises(ValueError):
        CapturedMedia(asset=asset)
    with pytest.raises(ValueError):
        CapturedMedia(asset=asset, content=b"", local_path=__file__)
