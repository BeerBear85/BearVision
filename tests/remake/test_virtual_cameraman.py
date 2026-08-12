import math

import pytest

from bearvision.processing import KalmanPositionTracker, VirtualCameramanConfig


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
