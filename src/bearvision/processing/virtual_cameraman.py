"""Coarse virtual cameraman with explicit position uncertainty.

The active BearVision 3 runtime historically stopped after clip extraction.  This
module provides the missing Edge-side post-processing slice for recorded-video
scenarios: run person detection over the extracted clip, estimate the rider's
image position with a forward Kalman pass plus Rauch--Tung--Striebel backward
smoothing, and generate a separate zero-phase camera path before cropping.

Green boxes are detector measurements.  The red cross and circle are the
smoothed rider estimate and a conservative circular 95 % confidence region
derived from the 2D position covariance.  The cyan rectangle follows the
separately smoothed virtual-camera path and is the actual crop window.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

from bearvision.config import VirtualCameramanConfig
from bearvision.contracts import BoundingBox, MediaAsset, PersonDetection
from bearvision.ports import CapturedMedia, Detector, InvalidComponentData, VideoFrame


CHI_SQUARE_2D_95 = 5.991464547107979


@dataclass(frozen=True, slots=True)
class ClipLengthAdjustment:
    """Frame window selected around the rider's in-frame trajectory."""

    source_start_frame_idx: int
    source_end_frame_idx_exclusive: int
    first_visible_frame_idx: int
    last_visible_frame_idx: int
    source_frame_count: int
    fps: float
    padding_s: float

    @property
    def source_start_s(self) -> float:
        return self.source_start_frame_idx / self.fps

    @property
    def source_end_s(self) -> float:
        return self.source_end_frame_idx_exclusive / self.fps

    @property
    def output_duration_s(self) -> float:
        return (
            self.source_end_frame_idx_exclusive - self.source_start_frame_idx
        ) / self.fps

    @property
    def source_duration_s(self) -> float:
        return self.source_frame_count / self.fps

    @property
    def adjusted(self) -> bool:
        return (
            self.source_start_frame_idx > 0
            or self.source_end_frame_idx_exclusive < self.source_frame_count
        )

    def to_dict(self) -> dict[str, int | float | bool]:
        return {
            "padding_s": self.padding_s,
            "source_start_frame_idx": self.source_start_frame_idx,
            "source_end_frame_idx_exclusive": self.source_end_frame_idx_exclusive,
            "first_visible_frame_idx": self.first_visible_frame_idx,
            "last_visible_frame_idx": self.last_visible_frame_idx,
            "source_start_s": self.source_start_s,
            "source_end_s": self.source_end_s,
            "source_duration_s": self.source_duration_s,
            "output_duration_s": self.output_duration_s,
            "adjusted": self.adjusted,
        }

def calculate_length_adjustment(
    rider_positions: Sequence[tuple[float, float]],
    *,
    frame_width_px: int,
    frame_height_px: int,
    fps: float,
    padding_s: float = 1.0,
) -> ClipLengthAdjustment:
    """Keep at most ``padding_s`` before and after the in-frame rider path."""

    if not rider_positions:
        raise ValueError("rider positions must not be empty")
    if frame_width_px <= 0 or frame_height_px <= 0:
        raise ValueError("frame dimensions must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if padding_s < 0:
        raise ValueError("padding must not be negative")

    visible = [
        index
        for index, (x_px, y_px) in enumerate(rider_positions)
        if 0.0 <= x_px < frame_width_px and 0.0 <= y_px < frame_height_px
    ]
    if not visible:
        raise InvalidComponentData("estimated rider position never enters the image")

    # Never retain more than the configured padding because of frame rounding.
    padding_frames = math.floor(padding_s * fps)
    first_visible = visible[0]
    last_visible = visible[-1]
    return ClipLengthAdjustment(
        source_start_frame_idx=max(0, first_visible - padding_frames),
        source_end_frame_idx_exclusive=min(
            len(rider_positions), last_visible + 1 + padding_frames
        ),
        first_visible_frame_idx=first_visible,
        last_visible_frame_idx=last_visible,
        source_frame_count=len(rider_positions),
        fps=fps,
        padding_s=padding_s,
    )


@dataclass(frozen=True, slots=True)
class TrackingFrame:
    frame_idx: int
    at_s: float
    source_frame_idx: int
    source_at_s: float
    estimate_x_px: float
    estimate_y_px: float
    confidence_radius_95_px: float
    covariance_xx_px2: float
    covariance_xy_px2: float
    covariance_yy_px2: float
    camera_x_px: float
    camera_y_px: float
    crop_box: BoundingBox
    detection: PersonDetection | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "at_s": self.at_s,
            "source_frame_idx": self.source_frame_idx,
            "source_at_s": self.source_at_s,
            "estimate": {
                "x_px": self.estimate_x_px,
                "y_px": self.estimate_y_px,
            },
            "confidence_radius_95_px": self.confidence_radius_95_px,
            "position_covariance_px2": [
                [self.covariance_xx_px2, self.covariance_xy_px2],
                [self.covariance_xy_px2, self.covariance_yy_px2],
            ],
            "camera_center": {
                "x_px": self.camera_x_px,
                "y_px": self.camera_y_px,
            },
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
    length_adjustment: ClipLengthAdjustment

    @property
    def reduction_ratio(self) -> float:
        if self.source_size_bytes == 0:
            return 0.0
        return 1.0 - self.processed_size_bytes / self.source_size_bytes


@dataclass(frozen=True, slots=True)
class PositionMeasurement:
    """One candidate image-position observation for the offline smoother."""

    x_px: float
    y_px: float
    standard_deviation_px: float
    confidence: float = 1.0
    detection: PersonDetection | None = None


@dataclass(frozen=True, slots=True)
class SmoothedPosition:
    """RTS-smoothed state and covariance for one video frame."""

    state: Any
    covariance: Any
    measurement: PositionMeasurement | None


class KalmanPositionTracker:
    """Damped constant-velocity 2D Kalman tracker with covariance output."""

    def __init__(
        self,
        *,
        process_noise_acceleration_px_s2: float = 45.0,
        initial_position_std_px: float = 8.0,
        initial_velocity_std_px_s: float = 80.0,
        velocity_damping_time_constant_s: float = 2.0,
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

    def model(self, dt_s: float) -> tuple[Any, Any]:
        """Return the transition and process covariance for one time step."""
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
        return transition, process_covariance

    def initialize(self, x_px: float, y_px: float, measurement_std_px: float) -> None:
        np = self._np
        position_variance = max(measurement_std_px, self.initial_position_std_px) ** 2
        velocity_variance = self.initial_velocity_std_px_s**2
        self.state = np.array([[x_px], [y_px], [0.0], [0.0]], dtype=float)
        self.covariance = np.diag(
            [position_variance, position_variance, velocity_variance, velocity_variance]
        )
        self.initialized = True

    def initialize_kinematics(
        self,
        x_px: float,
        y_px: float,
        velocity_x_px_s: float,
        velocity_y_px_s: float,
        measurement_std_px: float,
        velocity_std_px_s: float,
    ) -> None:
        """Initialize position and velocity from two plausible measurements."""
        self.initialize(x_px, y_px, measurement_std_px)
        self.state[2, 0] = velocity_x_px_s
        self.state[3, 0] = velocity_y_px_s
        velocity_variance = max(velocity_std_px_s, 1.0) ** 2
        self.covariance[2, 2] = velocity_variance
        self.covariance[3, 3] = velocity_variance

    def predict(self, dt_s: float) -> tuple[float, float]:
        if not self.initialized:
            raise RuntimeError("tracker must be initialized before predict")
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        transition, process_covariance = self.model(dt_s)
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_covariance
        return self.position

    def innovation_distance_squared(
        self, x_px: float, y_px: float, measurement_std_px: float
    ) -> float:
        """Return the normalized innovation squared for robust gating."""
        if not self.initialized:
            return 0.0
        np = self._np
        observation = np.array([[x_px], [y_px]], dtype=float)
        model = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        measurement_covariance = np.eye(2, dtype=float) * measurement_std_px**2
        innovation = observation - model @ self.state
        innovation_covariance = model @ self.covariance @ model.T + measurement_covariance
        return float(
            (innovation.T @ np.linalg.solve(innovation_covariance, innovation)).item()
        )

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


class KalmanRtsSmoother:
    """Offline 2D Kalman filter plus Rauch--Tung--Striebel backward pass."""

    def __init__(
        self,
        *,
        process_noise_acceleration_px_s2: float = 45.0,
        innovation_gate_chi2: float = 9.210340371976184,
        maximum_bootstrap_speed_px_s: float = 3_000.0,
        velocity_damping_time_constant_s: float = 2.0,
    ) -> None:
        if innovation_gate_chi2 <= 0:
            raise ValueError("innovation gate must be positive")
        if maximum_bootstrap_speed_px_s <= 0:
            raise ValueError("maximum bootstrap speed must be positive")
        if velocity_damping_time_constant_s <= 0:
            raise ValueError("velocity damping time constant must be positive")
        self.process_noise_acceleration_px_s2 = process_noise_acceleration_px_s2
        self.innovation_gate_chi2 = innovation_gate_chi2
        self.maximum_bootstrap_speed_px_s = maximum_bootstrap_speed_px_s
        self.velocity_damping_time_constant_s = velocity_damping_time_constant_s

    def smooth(
        self,
        frame_count: int,
        dt_s: float,
        measurements_by_frame: dict[int, tuple[PositionMeasurement, ...]],
    ) -> tuple[SmoothedPosition, ...]:
        import numpy as np

        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        available = sorted(index for index, values in measurements_by_frame.items() if values)
        if not available:
            raise ValueError("at least one measurement is required")
        first_index = available[0]
        first = max(measurements_by_frame[first_index], key=lambda item: item.confidence)
        tracker = KalmanPositionTracker(
            process_noise_acceleration_px_s2=self.process_noise_acceleration_px_s2,
            velocity_damping_time_constant_s=self.velocity_damping_time_constant_s,
        )
        tracker.initialize(first.x_px, first.y_px, first.standard_deviation_px)

        filtered_states: list[Any | None] = [None] * frame_count
        filtered_covariances: list[Any | None] = [None] * frame_count
        predicted_states: list[Any | None] = [None] * frame_count
        predicted_covariances: list[Any | None] = [None] * frame_count
        transitions: list[Any | None] = [None] * frame_count
        selected: list[PositionMeasurement | None] = [None] * frame_count
        filtered_states[first_index] = tracker.state.copy()
        filtered_covariances[first_index] = tracker.covariance.copy()
        selected[first_index] = first
        accepted_measurement_count = 1
        last_measurement = first
        last_measurement_index = first_index

        for index in range(first_index + 1, frame_count):
            transition, _ = tracker.model(dt_s)
            tracker.predict(dt_s)
            transitions[index] = transition
            predicted_states[index] = tracker.state.copy()
            predicted_covariances[index] = tracker.covariance.copy()
            candidates = measurements_by_frame.get(index, ())
            if candidates and accepted_measurement_count == 1:
                elapsed_s = (index - first_index) * dt_s
                plausible_bootstrap = [
                    candidate
                    for candidate in candidates
                    if math.hypot(
                        candidate.x_px - first.x_px,
                        candidate.y_px - first.y_px,
                    ) / elapsed_s <= self.maximum_bootstrap_speed_px_s
                ]
                if plausible_bootstrap:
                    measurement = max(
                        plausible_bootstrap, key=lambda item: item.confidence
                    )
                    velocity_x = (measurement.x_px - first.x_px) / elapsed_s
                    velocity_y = (measurement.y_px - first.y_px) / elapsed_s
                    velocity_std = math.hypot(
                        first.standard_deviation_px,
                        measurement.standard_deviation_px,
                    ) / elapsed_s
                    tracker.initialize_kinematics(
                        measurement.x_px,
                        measurement.y_px,
                        velocity_x,
                        velocity_y,
                        measurement.standard_deviation_px,
                        velocity_std,
                    )
                    selected[index] = measurement
                    accepted_measurement_count += 1
                    last_measurement = measurement
                    last_measurement_index = index
            else:
                accepted: list[tuple[float, PositionMeasurement]] = []
                for candidate in candidates:
                    distance = tracker.innovation_distance_squared(
                        candidate.x_px,
                        candidate.y_px,
                        candidate.standard_deviation_px,
                    )
                    if distance <= self.innovation_gate_chi2:
                        accepted.append((distance, candidate))
                if accepted:
                    _, measurement = min(
                        accepted, key=lambda item: (item[0], -item[1].confidence)
                    )
                    tracker.update(
                        measurement.x_px,
                        measurement.y_px,
                        measurement.standard_deviation_px,
                    )
                    selected[index] = measurement
                    accepted_measurement_count += 1
                    last_measurement = measurement
                    last_measurement_index = index
                elif candidates:
                    elapsed_s = (index - last_measurement_index) * dt_s
                    plausible_reacquisition = [
                        candidate
                        for candidate in candidates
                        if math.hypot(
                            candidate.x_px - last_measurement.x_px,
                            candidate.y_px - last_measurement.y_px,
                        ) / elapsed_s <= self.maximum_bootstrap_speed_px_s
                    ]
                    if plausible_reacquisition:
                        measurement = min(
                            plausible_reacquisition,
                            key=lambda item: (
                                math.hypot(
                                    item.x_px - tracker.position[0],
                                    item.y_px - tracker.position[1],
                                ),
                                -item.confidence,
                            ),
                        )
                        tracker.update(
                            measurement.x_px,
                            measurement.y_px,
                            measurement.standard_deviation_px,
                        )
                        selected[index] = measurement
                        accepted_measurement_count += 1
                        last_measurement = measurement
                        last_measurement_index = index
            filtered_states[index] = tracker.state.copy()
            filtered_covariances[index] = tracker.covariance.copy()

        smoothed_states = list(filtered_states)
        smoothed_covariances = list(filtered_covariances)
        for index in range(frame_count - 2, first_index - 1, -1):
            next_index = index + 1
            filtered_state = filtered_states[index]
            filtered_covariance = filtered_covariances[index]
            transition = transitions[next_index]
            predicted_state = predicted_states[next_index]
            predicted_covariance = predicted_covariances[next_index]
            next_state = smoothed_states[next_index]
            next_covariance = smoothed_covariances[next_index]
            if any(
                value is None
                for value in (
                    filtered_state,
                    filtered_covariance,
                    transition,
                    predicted_state,
                    predicted_covariance,
                    next_state,
                    next_covariance,
                )
            ):
                raise RuntimeError("incomplete Kalman history")
            assert filtered_state is not None
            assert filtered_covariance is not None
            assert transition is not None
            assert predicted_state is not None
            assert predicted_covariance is not None
            assert next_state is not None
            assert next_covariance is not None
            gain = filtered_covariance @ transition.T @ np.linalg.inv(predicted_covariance)
            smoothed_states[index] = filtered_state + gain @ (next_state - predicted_state)
            covariance = filtered_covariance + gain @ (
                next_covariance - predicted_covariance
            ) @ gain.T
            smoothed_covariances[index] = (covariance + covariance.T) / 2

        first_state = smoothed_states[first_index]
        first_covariance = smoothed_covariances[first_index]
        if first_state is None or first_covariance is None:
            raise RuntimeError("RTS smoother did not produce an initial state")
        for index in range(first_index - 1, -1, -1):
            elapsed = (first_index - index) * dt_s
            state = first_state.copy()
            state[0, 0] -= state[2, 0] * elapsed
            state[1, 0] -= state[3, 0] * elapsed
            covariance = first_covariance.copy()
            growth = (tracker.initial_velocity_std_px_s * elapsed) ** 2
            covariance[0, 0] += growth
            covariance[1, 1] += growth
            smoothed_states[index] = state
            smoothed_covariances[index] = covariance

        return tuple(
            SmoothedPosition(state, covariance, measurement)
            for state, covariance, measurement in zip(
                smoothed_states, smoothed_covariances, selected, strict=True
            )
        )


class ZeroPhaseButterworthCameraSmoother:
    """Second-order Butterworth run forward and backward for zero phase lag."""

    def __init__(self, cutoff_hz: float = 1.25) -> None:
        if cutoff_hz <= 0:
            raise ValueError("cutoff must be positive")
        self.cutoff_hz = cutoff_hz

    def smooth(self, values: list[float], sample_rate_hz: float) -> list[float]:
        import numpy as np

        if sample_rate_hz <= 0:
            raise ValueError("sample rate must be positive")
        if self.cutoff_hz >= sample_rate_hz / 2:
            raise ValueError("cutoff must be below Nyquist")
        if len(values) < 3:
            return list(values)
        omega = 2 * math.pi * self.cutoff_hz / sample_rate_hz
        cosine, sine = math.cos(omega), math.sin(omega)
        alpha = sine / math.sqrt(2.0)
        a0 = 1.0 + alpha
        b0 = (1.0 - cosine) / 2.0 / a0
        b1 = (1.0 - cosine) / a0
        b2 = b0
        a1 = -2.0 * cosine / a0
        a2 = (1.0 - alpha) / a0

        def forward(sequence: Any) -> Any:
            output = np.empty_like(sequence, dtype=float)
            x1 = x2 = y1 = y2 = float(sequence[0])
            for index, sample in enumerate(sequence):
                current = b0 * sample + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
                output[index] = current
                x2, x1 = x1, sample
                y2, y1 = y1, current
            return output

        data = np.asarray(values, dtype=float)
        pad = min(12, len(data) - 1)
        padded = np.pad(data, (pad, pad), mode="reflect")
        filtered = forward(forward(padded)[::-1])[::-1]
        return [float(value) for value in filtered[pad:-pad]]


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
        detections_by_frame: dict[int, tuple[PersonDetection, ...]] = {}
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
                        detections_by_frame[frame_idx] = tuple(detections)
                frame_idx += 1
        finally:
            capture.release()
        if not frames:
            raise InvalidComponentData("captured clip contains no frames")
        if not detections_by_frame:
            raise InvalidComponentData("virtual cameraman found no person in captured clip")

        measurements_by_frame = {
            index: tuple(
                PositionMeasurement(
                    *self._center(detection.bounding_box),
                    self._measurement_std(detection),
                    confidence=detection.confidence,
                    detection=detection,
                )
                for detection in detections
            )
            for index, detections in detections_by_frame.items()
        }
        smoother = KalmanRtsSmoother(
            process_noise_acceleration_px_s2=self.config.process_noise_acceleration_px_s2,
            innovation_gate_chi2=self.config.innovation_gate_chi2,
            maximum_bootstrap_speed_px_s=self.config.maximum_bootstrap_speed_px_s,
            velocity_damping_time_constant_s=(
                self.config.velocity_damping_time_constant_s
            ),
        )
        smoothed = smoother.smooth(len(frames), 1.0 / fps, measurements_by_frame)
        length_adjustment = calculate_length_adjustment(
            [(float(item.state[0, 0]), float(item.state[1, 0])) for item in smoothed],
            frame_width_px=width,
            frame_height_px=height,
            fps=fps,
            padding_s=self.config.length_adjustment_padding_s,
        )
        frames = frames[
            length_adjustment.source_start_frame_idx:
            length_adjustment.source_end_frame_idx_exclusive
        ]
        smoothed = smoothed[
            length_adjustment.source_start_frame_idx:
            length_adjustment.source_end_frame_idx_exclusive
        ]
        rider_x = [float(item.state[0, 0]) for item in smoothed]
        rider_y = [float(item.state[1, 0]) for item in smoothed]
        camera_smoother = ZeroPhaseButterworthCameraSmoother(self.config.camera_cutoff_hz)
        camera_x = camera_smoother.smooth(rider_x, fps)
        camera_y = camera_smoother.smooth(rider_y, fps)

        crop_width, crop_height = self._crop_dimensions(width, height)
        tracking: list[TrackingFrame] = []
        for index, item in enumerate(smoothed):
            source_index = length_adjustment.source_start_frame_idx + index
            x_px, y_px = float(item.state[0, 0]), float(item.state[1, 0])
            x_px = min(float(width), max(0.0, x_px))
            y_px = min(float(height), max(0.0, y_px))
            covariance = item.covariance[:2, :2]
            eigenvalues = self._position_covariance_eigenvalues(covariance)
            confidence_radius = math.sqrt(
                CHI_SQUARE_2D_95 * max(0.0, eigenvalues[-1])
            )
            crop = self._crop_box(
                camera_x[index], camera_y[index], crop_width, crop_height, width, height
            )
            tracking.append(
                TrackingFrame(
                    frame_idx=index,
                    at_s=index / fps,
                    source_frame_idx=source_index,
                    source_at_s=source_index / fps,
                    estimate_x_px=x_px,
                    estimate_y_px=y_px,
                    confidence_radius_95_px=confidence_radius,
                    covariance_xx_px2=float(covariance[0, 0]),
                    covariance_xy_px2=float(covariance[0, 1]),
                    covariance_yy_px2=float(covariance[1, 1]),
                    camera_x_px=camera_x[index],
                    camera_y_px=camera_y[index],
                    crop_box=crop,
                    detection=item.measurement.detection if item.measurement else None,
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
            await asyncio.to_thread(
                self._encode_video,
                silent_crop,
                processed_path,
            )
            await asyncio.to_thread(
                self._encode_video,
                silent_debug,
                debug_path,
            )
        finally:
            silent_crop.unlink(missing_ok=True)
            silent_debug.unlink(missing_ok=True)

        metadata = {
            "tracking_schema_version": "2.0",
            "source_filename": source.name,
            "processed_filename": processed_path.name,
            "debug_video_filename": debug_path.name,
            "coordinate_space": {"width_px": width, "height_px": height, "fps": fps},
            "state_estimator": "damped_constant_velocity_2d_kalman_plus_rts_smoother",
            "measurement_association": {
                "method": "nearest_normalized_innovation",
                "chi_square_gate": self.config.innovation_gate_chi2,
            },
            "confidence_region": "conservative circle using sqrt(chi2_2d_0.95 * max_eigenvalue(Pxy))",
            "camera_path": {
                "method": "second_order_butterworth_forward_backward",
                "cutoff_hz": self.config.camera_cutoff_hz,
                "zero_phase": True,
            },
            "length_adjustment": length_adjustment.to_dict(),
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
            length_adjustment=length_adjustment,
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

    @staticmethod
    def _position_covariance_eigenvalues(covariance: Any) -> list[float]:
        import numpy as np

        return [float(value) for value in np.linalg.eigvalsh(covariance)]

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

    def _encode_video(
        self,
        video_path: Path,
        destination: Path,
    ) -> None:
        partial = destination.with_name(f"{destination.stem}.partial{destination.suffix}")
        partial.unlink(missing_ok=True)
        command = [
            self.ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video_path), "-map", "0:v:0", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", str(self.config.output_crf),
            "-an", "-movflags", "+faststart", str(partial),
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
