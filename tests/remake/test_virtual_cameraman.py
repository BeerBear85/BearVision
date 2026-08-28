import asyncio
from datetime import datetime, timezone
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from bearvision.config import VirtualCameramanConfig
from bearvision.contracts import MediaAsset
from bearvision.processing.virtual_cameraman import (
    KalmanPositionTracker,
    KalmanRtsSmoother,
    PositionMeasurement,
    VirtualCameramanProcessor,
    ZeroPhaseButterworthCameraSmoother,
    calculate_length_adjustment,
)
from bearvision.ports import CapturedMedia, ClipProcessor


def test_virtual_cameraman_owns_prepared_clip_mapping_at_clip_processor_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = CapturedMedia(
        asset=MediaAsset(
            asset_id="source-1",
            filename="source.mp4",
            content_type="video/mp4",
            size_bytes=1_000,
            created_at_utc=datetime.now(timezone.utc),
        ),
        local_path=tmp_path / "source.mp4",
    )
    processed_media = CapturedMedia(
        asset=MediaAsset(
            asset_id="processed-source-1",
            filename="source.virtual-cameraman.mp4",
            content_type="video/mp4",
            size_bytes=400,
            created_at_utc=datetime.now(timezone.utc),
        ),
        local_path=tmp_path / "source.virtual-cameraman.mp4",
    )
    adjustment = calculate_length_adjustment(
        [(-1.0, 5.0), (5.0, 5.0), (11.0, 5.0)],
        frame_width_px=10,
        frame_height_px=10,
        fps=1.0,
        padding_s=0,
    )
    internal_result = SimpleNamespace(
        media=processed_media,
        metadata_path=tmp_path / "source.tracking.json",
        debug_video_path=tmp_path / "source.tracking-debug.mp4",
        source_size_bytes=1_000,
        processed_size_bytes=400,
        reduction_ratio=0.6,
        source_width_px=1920,
        source_height_px=1080,
        tracking_frames=(),
        length_adjustment=adjustment,
    )

    processor = VirtualCameramanProcessor(object(), object(), tmp_path)

    async def process_media(media: CapturedMedia):
        assert media is source
        return internal_result

    monkeypatch.setattr(processor, "_process_media", process_media)

    prepared = asyncio.run(processor.process(source))

    assert isinstance(processor, ClipProcessor)
    assert prepared.media is processed_media
    assert prepared.source_start_offset_s == 1.0
    assert prepared.duration_s == 1.0
    assert [event.kind for event in prepared.trace_events] == [
        "virtual_cameraman_completed"
    ]
    assert prepared.trace_events[0].payload["size_reduction_ratio"] == 0.6


def test_kalman_tracker_reports_finite_95_percent_position_region() -> None:
    tracker = KalmanPositionTracker(process_noise_acceleration_px_s2=10.0)
    tracker.initialize(100.0, 50.0, measurement_std_px=3.0)

    initial_radius = tracker.confidence_radius_95_px
    tracker.predict(0.1)
    predicted_radius = tracker.confidence_radius_95_px
    tracker.update(103.0, 51.0, measurement_std_px=2.0)
    corrected_radius = tracker.confidence_radius_95_px

    assert tracker.position[0] == pytest.approx(103.0, abs=1.0)
    assert tracker.position[1] == pytest.approx(51.0, abs=1.0)
    assert math.isfinite(corrected_radius)
    assert predicted_radius > initial_radius
    assert corrected_radius < predicted_radius


def test_virtual_cameraman_output_requires_even_h264_dimensions() -> None:
    with pytest.raises(ValueError, match="must be even"):
        VirtualCameramanConfig(output_width_px=161)


def test_length_adjustment_keeps_one_second_around_in_frame_trajectory() -> None:
    positions = [(-1.0, 50.0)] * 20 + [(50.0, 50.0)] * 50 + [(101.0, 50.0)] * 30

    adjustment = calculate_length_adjustment(
        positions,
        frame_width_px=100,
        frame_height_px=100,
        fps=10.0,
    )

    assert adjustment.source_start_frame_idx == 10
    assert adjustment.source_end_frame_idx_exclusive == 80
    assert adjustment.source_start_s == pytest.approx(1.0)
    assert adjustment.source_end_s == pytest.approx(8.0)
    assert adjustment.output_duration_s == pytest.approx(7.0)
    assert adjustment.adjusted


def test_length_adjustment_does_not_extend_past_source_boundaries() -> None:
    adjustment = calculate_length_adjustment(
        [(50.0, 50.0)] * 20,
        frame_width_px=100,
        frame_height_px=100,
        fps=10.0,
    )

    assert adjustment.source_start_frame_idx == 0
    assert adjustment.source_end_frame_idx_exclusive == 20
    assert adjustment.output_duration_s == pytest.approx(2.0)
    assert not adjustment.adjusted


def test_rts_smoother_uses_future_measurement_to_improve_past_state() -> None:
    smoother = KalmanRtsSmoother(
        process_noise_acceleration_px_s2=5.0,
        innovation_gate_chi2=100.0,
    )
    trajectory = smoother.smooth(
        frame_count=5,
        dt_s=1.0,
        measurements_by_frame={
            0: (PositionMeasurement(0.0, 0.0, 1.0),),
            4: (PositionMeasurement(40.0, 0.0, 1.0),),
        },
    )

    assert trajectory[2].state[0, 0] > 1.0
    assert trajectory[2].state[0, 0] < trajectory[4].state[0, 0]
    assert trajectory[2].covariance[0, 0] < 100.0


def test_rts_smoother_rejects_implausible_measurement() -> None:
    smoother = KalmanRtsSmoother(
        process_noise_acceleration_px_s2=2.0,
        innovation_gate_chi2=9.210340371976184,
    )
    trajectory = smoother.smooth(
        frame_count=3,
        dt_s=0.1,
        measurements_by_frame={
            0: (PositionMeasurement(10.0, 10.0, 1.0),),
            1: (PositionMeasurement(10_000.0, 10_000.0, 1.0),),
        },
    )

    assert trajectory[1].measurement is None
    assert trajectory[1].state[0, 0] < 100.0


def test_rts_smoother_bootstraps_fast_plausible_rider_velocity() -> None:
    smoother = KalmanRtsSmoother(
        process_noise_acceleration_px_s2=45.0,
        innovation_gate_chi2=9.210340371976184,
    )
    trajectory = smoother.smooth(
        frame_count=19,
        dt_s=1 / 60,
        measurements_by_frame={
            0: (PositionMeasurement(121.5, 716.0, 12.0),),
            6: (PositionMeasurement(243.5, 711.5, 12.0),),
            12: (PositionMeasurement(352.0, 698.0, 12.0),),
            18: (PositionMeasurement(433.0, 688.0, 12.0),),
        },
    )

    assert trajectory[6].measurement is not None
    assert trajectory[12].measurement is not None
    assert trajectory[18].measurement is not None
    assert trajectory[18].state[0, 0] == pytest.approx(433.0, abs=20.0)


def test_rts_smoother_reacquires_plausible_rider_after_detection_gap() -> None:
    smoother = KalmanRtsSmoother(maximum_bootstrap_speed_px_s=1_000.0)
    trajectory = smoother.smooth(
        frame_count=6,
        dt_s=0.1,
        measurements_by_frame={
            0: (PositionMeasurement(0.0, 0.0, 2.0),),
            1: (PositionMeasurement(50.0, 0.0, 2.0),),
            5: (PositionMeasurement(250.0, 100.0, 2.0),),
        },
    )

    assert trajectory[5].measurement is not None
    assert trajectory[5].state[0, 0] > 200.0


def test_camera_smoother_is_zero_phase_and_reduces_jitter() -> None:
    smoother = ZeroPhaseButterworthCameraSmoother(cutoff_hz=1.0)
    impulse = [0.0] * 101
    impulse[50] = 1.0
    filtered_impulse = smoother.smooth(impulse, sample_rate_hz=30.0)
    jitter = [float(index % 2) for index in range(100)]
    filtered_jitter = smoother.smooth(jitter, sample_rate_hz=30.0)

    assert max(range(len(filtered_impulse)), key=filtered_impulse.__getitem__) == 50
    assert filtered_impulse[45] == pytest.approx(filtered_impulse[55], abs=1e-6)
    raw_variation = sum(abs(b - a) for a, b in zip(jitter, jitter[1:]))
    filtered_variation = sum(
        abs(b - a) for a, b in zip(filtered_jitter, filtered_jitter[1:])
    )
    assert filtered_variation < raw_variation * 0.1
