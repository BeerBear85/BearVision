"""Reusable behavioural checks for real and simulated port implementations."""

from __future__ import annotations

from bearvision.contracts import CaptureRequest
from bearvision.ports import Camera, CapturedMedia, Clock, Detector, Storage, TagRegistry, TagScanner, VideoFrame


async def check_clock(clock: Clock) -> None:
    before = clock.monotonic()
    await clock.sleep(0.25)
    after = clock.monotonic()
    assert after >= before + 0.25
    assert clock.utc_now().tzinfo is not None
    assert clock.utc_now().utcoffset() is not None


async def check_camera(camera: Camera, request: CaptureRequest) -> CapturedMedia:
    await camera.connect()
    preview = await camera.start_preview()
    assert preview
    first = await camera.capture(request)
    second = await camera.capture(request)
    assert first == second, "capture must be idempotent for a stable request_id"
    assert first.asset.asset_id
    await camera.stop_preview()
    await camera.disconnect()
    return first


async def check_scanner(scanner: TagScanner) -> None:
    observations = [item async for item in scanner.observations()]
    assert observations
    assert all(item.observed_at_utc.tzinfo is not None for item in observations)


async def check_detector(detector: Detector, frame: VideoFrame) -> None:
    detections = await detector.detect(frame)
    assert isinstance(detections, tuple)
    assert all(item.frame_id == frame.frame_id for item in detections)


async def check_storage(storage: Storage, media: CapturedMedia, object_key: str) -> None:
    first = await storage.upload(media, object_key)
    second = await storage.upload(media, object_key)
    assert first == second, "upload must be idempotent for the same asset and object key"
    assert first.asset_id == media.asset.asset_id
    assert await storage.download(object_key)
    await storage.delete(object_key)


def check_registry(registry: TagRegistry, known_tag_id: str) -> None:
    entry = registry.resolve(known_tag_id)
    assert entry is not None
    assert entry.tag_id == known_tag_id
    assert registry.resolve("missing-tag") is None
    assert entry in tuple(registry.entries())
