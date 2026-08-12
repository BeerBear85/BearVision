import asyncio
import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from bearvision.contracts import (
    BoundingBox,
    CaptureRequest,
    MediaAsset,
    PersonDetection,
    RiderAssignment,
    RiderAssignmentStatus,
    StorageReceipt,
    TagObservation,
    TagRegistryEntry,
    Vector3,
)
from bearvision.ports import (
    Camera,
    CapturedMedia,
    Clock,
    Detector,
    Storage,
    TagRegistry,
    TagScanner,
    VideoFrame,
)
from bearvision.testing import (
    check_camera,
    check_clock,
    check_detector,
    check_registry,
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
        self.captures: dict[str, CapturedMedia] = {}

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def start_preview(self) -> str:
        return "memory://preview"

    async def stop_preview(self) -> None:
        return None

    async def capture(self, request: CaptureRequest) -> CapturedMedia:
        if request.request_id not in self.captures:
            content = b"reference-video"
            self.captures[request.request_id] = CapturedMedia(
                asset=MediaAsset(
                    asset_id=f"asset-{request.request_id}",
                    filename=f"{request.request_id}.mp4",
                    content_type="video/mp4",
                    size_bytes=len(content),
                    created_at_utc=NOW,
                ),
                content=content,
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


class ReferenceRegistry:
    def __init__(self, entry: TagRegistryEntry) -> None:
        self.entry = entry

    def resolve(self, tag_id: str) -> TagRegistryEntry | None:
        return self.entry if self.entry.tag_id == tag_id and self.entry.enabled else None

    def entries(self):
        return (self.entry,)


def assignment() -> RiderAssignment:
    return RiderAssignment(
        status=RiderAssignmentStatus.ASSIGNED,
        assigned_at_monotonic_s=1,
        rider_id="rider-17",
        tag_id="tag-17",
        candidate_tag_ids=("tag-17",),
        reason="one registered tag qualifies",
    )


def request() -> CaptureRequest:
    return CaptureRequest(
        request_id="capture-1",
        requested_at_monotonic_s=1,
        pre_roll_s=15,
        post_roll_s=5,
        assignment=assignment(),
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
    assert isinstance(ReferenceRegistry(TagRegistryEntry(tag_id="tag-17", rider_id="rider-17")), TagRegistry)


def test_reusable_component_contract_suites() -> None:
    clock = ReferenceClock()
    camera = ReferenceCamera()
    scanner = ReferenceScanner((observation(),))
    detector = ReferenceDetector()
    storage = ReferenceStorage()
    registry = ReferenceRegistry(TagRegistryEntry(tag_id="tag-17", rider_id="rider-17"))
    frame = VideoFrame("frame-1", 1, 1920, 1080, b"pixels")

    asyncio.run(check_clock(clock))
    media = asyncio.run(check_camera(camera, request()))
    asyncio.run(check_scanner(scanner))
    asyncio.run(check_detector(detector, frame))
    asyncio.run(check_storage(storage, media, "rider-17/capture-1.mp4"))
    check_registry(registry, "tag-17")


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
