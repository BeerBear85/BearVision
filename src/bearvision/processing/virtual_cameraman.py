"""Coarse virtual cameraman with explicit position uncertainty.

The active BearVision 3 runtime historically stopped after clip extraction.  This
module provides the missing Edge-side post-processing slice for recorded-video
scenarios: run person detection over the extracted clip, estimate the rider's
image position with a damped constant-velocity Kalman filter, crop around that estimate,
and export metadata plus an annotated engineering preview.

Green boxes are detector measurements.  The red cross and circle are the Kalman
position estimate and a conservative circular 95 % confidence region derived
from the 2D position covariance.  The cyan rectangle is the actual crop window.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

from bearvision.contracts import BoundingBox, MediaAsset, PersonDetection
from bearvision.ports import CapturedMedia, Detector, InvalidComponentData, VideoFrame


CHI_SQUARE_2D_95 = 5.991464547107979


@dataclass(frozen=True, slots=True)
class VirtualCameramanConfig:
    """Version-local policy for the coarse crop pass."""

    sample_fps: float = 10.0
    crop_width_ratio: float = 0.5
    output_width_px: int = 160
    output_height_px: int = 90
    process_noise_acceleration_px_s2: float = 45.0
    minimum_measurement_std_px: float = 2.0
    output_crf: int = 24

    def __post_init__(self) -> None:
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if not 0 < self.crop_width_ratio <= 1:
            raise ValueError("crop_width_ratio must be in (0, 1]")
        if self.output_width_px <= 0 or self.output_height_px <= 0:
            raise ValueError("output dimensions must be positive")
        if self.output_width_px % 2 or self.output_height_px % 2:
            raise ValueError("H.264 output dimensions must be even")
        if self.process_noise_acceleration_px_s2 <= 0:
            raise ValueError("process noise must be positive")
        if self.minimum_measurement_std_px <= 0:
            raise ValueError("measurement standard deviation must be positive")


@dataclass(frozen=True, slots=True)
class TrackingFrame:
    frame_idx: int
    at_s: float
    estimate_x_px: float
    estimate_y_px: float
    confidence_radius_95_px: float
    covariance_xx_px2: float
    covariance_xy_px2: float
    covariance_yy_px2: float
    crop_box: BoundingBox
    detection: PersonDetection | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "at_s": self.at_s,
            "estimate": {
                "x_px": self.estimate_x_px,
                "y_px": self.estimate_y_px,
            },
            "confidence_radius_95_px": self.confidence_radius_95_px,
            "position_covariance_px2": [
                [self.covariance_xx_px2, self.covariance_xy_px2],
                [self.covariance_xy_px2, self.covariance_yy_px2],
            ],
            "crop_box": self.crop_box.model_dump(mode="json"),
            "detection": self.detection.model_dump(mode="json") if self.detection else None,
        }


@dataclass(frozen=True, slots=True)
class ProcessedClip:
    media: CapturedMedia
    metadata_path: Path
    debug_video_path: Path
    source_size_bytes: int
    processed_size_bytes: int
    source_width_px: int
    source_height_px: int
    tracking_frames: tuple[TrackingFrame, ...]

    @property
    def reduction_ratio(self) -> float:
        if self.source_size_bytes == 0:
            return 0.0
        return 1.0 - self.processed_size_bytes / self.source_size_bytes


class KalmanPositionTracker:
    """Damped constant-velocity 2D Kalman tracker with covariance output."""

    def __init__(
        self,
        *,
        process_noise_acceleration_px_s2: float = 45.0,
        initial_position_std_px: float = 8.0,
        initial_velocity_std_px_s: float = 80.0,
        velocity_damping_time_constant_s: float = 0.35,
    ) -> None:
        import numpy as np

        if process_noise_acceleration_px_s2 <= 0:
            raise ValueError("process noise must be positive")
        self._np = np
        self.process_noise_acceleration_px_s2 = process_noise_acceleration_px_s2
        self.initial_position_std_px = initial_position_std_px
        self.initial_velocity_std_px_s = initial_velocity_std_px_s
        if velocity_damping_time_constant_s <= 0:
            raise ValueError("velocity damping time constant must be positive")
        self.velocity_damping_time_constant_s = velocity_damping_time_constant_s
        self.state = np.zeros((4, 1), dtype=float)
        self.covariance = np.eye(4, dtype=float)
        self.initialized = False

    def initialize(self, x_px: float, y_px: float, measurement_std_px: float) -> None:
        np = self._np
        position_variance = max(measurement_std_px, self.initial_position_std_px) ** 2
        velocity_variance = self.initial_velocity_std_px_s**2
        self.state = np.array([[x_px], [y_px], [0.0], [0.0]], dtype=float)
        self.covariance = np.diag(
            [position_variance, position_variance, velocity_variance, velocity_variance]
        )
        self.initialized = True

    def predict(self, dt_s: float) -> tuple[float, float]:
        if not self.initialized:
            raise RuntimeError("tracker must be initialized before predict")
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        np = self._np
        damping = math.exp(-dt_s / self.velocity_damping_time_constant_s)
        transition = np.array(
            [[1.0, 0.0, dt_s, 0.0], [0.0, 1.0, 0.0, dt_s],
             [0.0, 0.0, damping, 0.0], [0.0, 0.0, 0.0, damping]],
            dtype=float,
        )
        dt2, dt3, dt4 = dt_s**2, dt_s**3, dt_s**4
        q = self.process_noise_acceleration_px_s2**2
        process_covariance = q * np.array(
            [[dt4 / 4, 0, dt3 / 2, 0], [0, dt4 / 4, 0, dt3 / 2],
             [dt3 / 2, 0, dt2, 0], [0, dt3 / 2, 0, dt2]],
            dtype=float,
        )
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_covariance
        return self.position

    def update(self, x_px: float, y_px: float, measurement_std_px: float) -> tuple[float, float]:
        np = self._np
        if not self.initialized:
            self.initialize(x_px, y_px, measurement_std_px)
            return self.position
        observation = np.array([[x_px], [y_px]], dtype=float)
        model = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        measurement_covariance = np.eye(2, dtype=float) * measurement_std_px**2
        innovation = observation - model @ self.state
        innovation_covariance = model @ self.covariance @ model.T + measurement_covariance
        gain = self.covariance @ model.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + gain @ innovation
        identity = np.eye(4, dtype=float)
        # Joseph form maintains a symmetric positive-semidefinite covariance.
        remainder = identity - gain @ model
        self.covariance = (
            remainder @ self.covariance @ remainder.T
            + gain @ measurement_covariance @ gain.T
        )
        return self.position

    @property
    def position(self) -> tuple[float, float]:
        return float(self.state[0, 0]), float(self.state[1, 0])

    @property
    def position_covariance(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (float(self.covariance[0, 0]), float(self.covariance[0, 1])),
            (float(self.covariance[1, 0]), float(self.covariance[1, 1])),
        )

    @property
    def confidence_radius_95_px(self) -> float:
        np = self._np
        eigenvalues = np.linalg.eigvalsh(self.covariance[:2, :2])
        return math.sqrt(CHI_SQUARE_2D_95 * max(0.0, float(eigenvalues[-1])))


class VirtualCameramanProcessor:
    """Track, annotate and crop one extracted local clip."""

    def __init__(
        self,
        detector: Detector,
        clock: Any,
        *,
        config: VirtualCameramanConfig | None = None,
        ffmpeg_path: str | Path = "ffmpeg",
    ) -> None:
        self.detector = detector
        self.clock = clock
        self.config = config or VirtualCameramanConfig()
        self.ffmpeg_path = str(ffmpeg_path)

    async def process(self, media: CapturedMedia, output_dir: str | Path) -> ProcessedClip:
        if media.local_path is None:
            raise InvalidComponentData("virtual cameraman requires file-backed media")
        source = media.local_path.resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - Edge optional dependency
            raise InvalidComponentData("OpenCV is required for virtual cameraman") from exc

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise InvalidComponentData(f"cannot open captured clip: {source}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0:
            capture.release()
            raise InvalidComponentData("captured clip reports invalid media metadata")

        sample_step = max(1, round(fps / self.config.sample_fps))
        frames: list[Any] = []
        detections_by_frame: dict[int, PersonDetection] = {}
        frame_idx = 0
        try:
            while True:
                ok, pixels = await asyncio.to_thread(capture.read)
                if not ok:
                    break
                frames.append(pixels)
                if frame_idx % sample_step == 0:
                    video_frame = VideoFrame(
                        frame_id=f"post-frame-{frame_idx}",
                        observed_at_monotonic_s=frame_idx / fps,
                        width_px=width,
                        height_px=height,
                        payload=pixels,
                    )
                    detections = await self.detector.detect(video_frame)
                    if detections:
                        detections_by_frame[frame_idx] = max(
                            detections, key=lambda item: item.confidence
                        )
                frame_idx += 1
        finally:
            capture.release()
        if not frames:
            raise InvalidComponentData("captured clip contains no frames")
        if not detections_by_frame:
            raise InvalidComponentData("virtual cameraman found no person in captured clip")

        first_detection = detections_by_frame[min(detections_by_frame)]
        tracker = KalmanPositionTracker(
            process_noise_acceleration_px_s2=self.config.process_noise_acceleration_px_s2
        )
        first_center = self._center(first_detection.bounding_box)
        first_std = self._measurement_std(first_detection)
        tracker.initialize(*first_center, first_std)

        crop_width, crop_height = self._crop_dimensions(width, height)
        tracking: list[TrackingFrame] = []
        for index in range(len(frames)):
            if index > 0:
                tracker.predict(1.0 / fps)
            detection = detections_by_frame.get(index)
            if detection is not None:
                tracker.update(*self._center(detection.bounding_box), self._measurement_std(detection))
            x_px, y_px = tracker.position
            x_px = min(float(width), max(0.0, x_px))
            y_px = min(float(height), max(0.0, y_px))
            covariance = tracker.position_covariance
            crop = self._crop_box(x_px, y_px, crop_width, crop_height, width, height)
            tracking.append(
                TrackingFrame(
                    frame_idx=index,
                    at_s=index / fps,
                    estimate_x_px=x_px,
                    estimate_y_px=y_px,
                    confidence_radius_95_px=tracker.confidence_radius_95_px,
                    covariance_xx_px2=covariance[0][0],
                    covariance_xy_px2=covariance[0][1],
                    covariance_yy_px2=covariance[1][1],
                    crop_box=crop,
                    detection=detection,
                )
            )

        stem = source.stem
        silent_crop = output_dir / f"{stem}.virtual-cameraman.silent.mp4"
        silent_debug = output_dir / f"{stem}.tracking-debug.silent.mp4"
        processed_path = output_dir / f"{stem}.virtual-cameraman.mp4"
        debug_path = output_dir / f"{stem}.tracking-debug.mp4"
        metadata_path = output_dir / f"{stem}.tracking.json"

        await asyncio.to_thread(
            self._render_videos,
            frames,
            tracking,
            fps,
            silent_crop,
            silent_debug,
            width,
            height,
        )
        try:
            await asyncio.to_thread(self._encode_with_source_audio, silent_crop, source, processed_path)
            await asyncio.to_thread(self._encode_with_source_audio, silent_debug, source, debug_path)
        finally:
            silent_crop.unlink(missing_ok=True)
            silent_debug.unlink(missing_ok=True)

        metadata = {
            "tracking_schema_version": "1.0",
            "source_filename": source.name,
            "processed_filename": processed_path.name,
            "debug_video_filename": debug_path.name,
            "coordinate_space": {"width_px": width, "height_px": height, "fps": fps},
            "kalman_model": "damped_constant_velocity_2d",
            "confidence_region": "conservative circle using sqrt(chi2_2d_0.95 * max_eigenvalue(Pxy))",
            "crop_output": {
                "width_px": self.config.output_width_px,
                "height_px": self.config.output_height_px,
                "source_crop_width_px": crop_width,
                "source_crop_height_px": crop_height,
            },
            "frames": [item.to_dict() for item in tracking],
        }
        temporary_metadata = metadata_path.with_suffix(".partial.json")
        temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(temporary_metadata, metadata_path)

        asset = MediaAsset(
            asset_id=f"processed-{media.asset.asset_id}",
            filename=processed_path.name,
            content_type="video/mp4",
            size_bytes=processed_path.stat().st_size,
            created_at_utc=self.clock.utc_now(),
        )
        return ProcessedClip(
            media=CapturedMedia(asset=asset, local_path=processed_path),
            metadata_path=metadata_path,
            debug_video_path=debug_path,
            source_size_bytes=source.stat().st_size,
            processed_size_bytes=processed_path.stat().st_size,
            source_width_px=width,
            source_height_px=height,
            tracking_frames=tuple(tracking),
        )

    def _measurement_std(self, detection: PersonDetection) -> float:
        size_term = min(
            detection.bounding_box.width_px, detection.bounding_box.height_px
        ) * 0.15
        return max(
            self.config.minimum_measurement_std_px,
            size_term / max(0.2, detection.confidence),
        )

    @staticmethod
    def _center(box: BoundingBox) -> tuple[float, float]:
        return box.x_px + box.width_px / 2, box.y_px + box.height_px / 2

    def _crop_dimensions(self, width: int, height: int) -> tuple[int, int]:
        crop_width = max(2, int(round(width * self.config.crop_width_ratio)))
        target_aspect = self.config.output_width_px / self.config.output_height_px
        crop_height = max(2, int(round(crop_width / target_aspect)))
        if crop_height > height:
            crop_height = height
            crop_width = int(round(crop_height * target_aspect))
        crop_width = min(width, crop_width)
        return crop_width, crop_height

    @staticmethod
    def _crop_box(
        center_x: float,
        center_y: float,
        crop_width: int,
        crop_height: int,
        frame_width: int,
        frame_height: int,
    ) -> BoundingBox:
        x = min(max(0.0, center_x - crop_width / 2), frame_width - crop_width)
        y = min(max(0.0, center_y - crop_height / 2), frame_height - crop_height)
        return BoundingBox(x_px=x, y_px=y, width_px=crop_width, height_px=crop_height)

    def _render_videos(
        self,
        frames: list[Any],
        tracking: list[TrackingFrame],
        fps: float,
        crop_path: Path,
        debug_path: Path,
        width: int,
        height: int,
    ) -> None:
        import cv2

        codec = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        crop_writer = cv2.VideoWriter(
            str(crop_path), codec, fps,
            (self.config.output_width_px, self.config.output_height_px),
        )
        debug_writer = cv2.VideoWriter(str(debug_path), codec, fps, (width, height))
        if not crop_writer.isOpened() or not debug_writer.isOpened():
            crop_writer.release()
            debug_writer.release()
            raise InvalidComponentData("cannot create virtual cameraman output")
        try:
            for pixels, item in zip(frames, tracking, strict=True):
                box = item.crop_box
                x1, y1 = int(round(box.x_px)), int(round(box.y_px))
                x2 = min(width, x1 + int(round(box.width_px)))
                y2 = min(height, y1 + int(round(box.height_px)))
                crop = pixels[y1:y2, x1:x2]
                crop_writer.write(
                    cv2.resize(
                        crop,
                        (self.config.output_width_px, self.config.output_height_px),
                        interpolation=cv2.INTER_AREA,
                    )
                )
                debug = pixels.copy()
                if item.detection is not None:
                    detected = item.detection.bounding_box
                    dx1, dy1 = int(detected.x_px), int(detected.y_px)
                    dx2 = int(detected.x_px + detected.width_px)
                    dy2 = int(detected.y_px + detected.height_px)
                    cv2.rectangle(debug, (dx1, dy1), (dx2, dy2), (50, 220, 80), 2)
                cx, cy = int(round(item.estimate_x_px)), int(round(item.estimate_y_px))
                radius = max(2, int(round(item.confidence_radius_95_px)))
                cv2.circle(debug, (cx, cy), radius, (40, 40, 230), 1)
                cv2.line(debug, (cx - 6, cy), (cx + 6, cy), (40, 40, 230), 2)
                cv2.line(debug, (cx, cy - 6), (cx, cy + 6), (40, 40, 230), 2)
                cv2.rectangle(debug, (x1, y1), (x2, y2), (230, 190, 40), 1)
                debug_writer.write(debug)
        finally:
            crop_writer.release()
            debug_writer.release()

    def _encode_with_source_audio(self, video_path: Path, source: Path, destination: Path) -> None:
        partial = destination.with_name(f"{destination.stem}.partial{destination.suffix}")
        partial.unlink(missing_ok=True)
        command = [
            self.ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video_path), "-i", str(source),
            "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", str(self.config.output_crf),
            "-c:a", "aac", "-movflags", "+faststart", "-shortest", str(partial),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            if completed.returncode != 0:
                raise InvalidComponentData(
                    "virtual cameraman encoding failed: "
                    + (completed.stderr.strip() or "FFmpeg failed")
                )
            os.replace(partial, destination)
        finally:
            partial.unlink(missing_ok=True)
