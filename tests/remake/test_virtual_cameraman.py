import math

import pytest

from bearvision.processing import (
    KalmanPositionTracker,
    KalmanRtsSmoother,
    PositionMeasurement,
    VirtualCameramanConfig,
    ZeroPhaseButterworthCameraSmoother,
)


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
