"""Reusable behavioural checks for real and simulated port implementations."""

from __future__ import annotations

from bearvision.contracts import CaptureRequest
from bearvision.ports import (
    Camera,
    CapturedClip,
    CapturedMedia,
    CaptureWindowPrecision,
    Clock,
    Detector,
    Storage,
    TagScanner,
    VideoFrame,
)


async def check_clock(clock: Clock) -> None:
    before = clock.monotonic()
    await clock.sleep(0.25)
    after = clock.monotonic()
    assert after >= before + 0.25
    assert clock.utc_now().tzinfo is not None
    assert clock.utc_now().utcoffset() is not None


async def check_camera(camera: Camera, request: CaptureRequest) -> CapturedClip:
    await camera.connect()
    preview = await camera.start_preview()
    assert preview
    first = await camera.capture(request)
    second = await camera.capture(request)
    assert first == second, "capture must be idempotent for a stable request_id"
    assert first.request_id == request.request_id
    assert first.media.asset.asset_id
    assert first.requested_window.start_monotonic_s >= max(
        0.0, request.requested_at_monotonic_s - request.pre_roll_s
    )
    assert first.requested_window.end_monotonic_s == (
        request.requested_at_monotonic_s + request.post_roll_s
    )
    assert first.actual_window.start_monotonic_s <= request.requested_at_monotonic_s
    assert first.actual_window.end_monotonic_s >= request.requested_at_monotonic_s
    assert isinstance(first.actual_window.precision, CaptureWindowPrecision)
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
