"""Tests for wakeboard cable-park simulation."""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.cable_path import CablePath
from sim.config import CABLE_TOWER_LOCATIONS, SimConfig


class TestCablePath:
    """Tests for CablePath class."""

    def test_path_length_positive(self):
        """Test that cable path has positive length."""
        path = CablePath(CABLE_TOWER_LOCATIONS)
        assert path.total_length > 0
        assert path.total_length > 100  # Should be hundreds of meters

    def test_wraparound_continuity(self):
        """Test that position at s=0 equals position at s=total_length."""
        path = CablePath(CABLE_TOWER_LOCATIONS)

        pos_0, _ = path.get_position_at_distance(0.0)
        pos_end, _ = path.get_position_at_distance(path.total_length)

        # Should be at same position (within numerical tolerance)
        np.testing.assert_allclose(pos_0, pos_end, rtol=1e-10)

    def test_position_on_path(self):
        """Test that positions are on the polyline."""
        path = CablePath(CABLE_TOWER_LOCATIONS)

        # Test several positions along path
        for s in np.linspace(0, path.total_length * 0.99, 20):
            pos, seg_idx = path.get_position_at_distance(s)

            # Check that we got valid segment index
            assert 0 <= seg_idx < len(CABLE_TOWER_LOCATIONS)

            # Check position is 2D
            assert pos.shape == (2,)

    def test_tangent_is_normalized(self):
        """Test that tangent vectors are unit vectors."""
        path = CablePath(CABLE_TOWER_LOCATIONS)

        for s in np.linspace(0, path.total_length * 0.99, 20):
            tangent = path.get_tangent_at_distance(s)

            # Should be unit vector
            length = np.linalg.norm(tangent)
            np.testing.assert_allclose(length, 1.0, rtol=1e-10)


class TestSimulationIntegration:
    """Integration tests for full simulation."""

    @pytest.fixture
    def temp_outdir(self, tmp_path):
        """Create temporary output directory."""
        outdir = tmp_path / "sim_output"
        outdir.mkdir()
        return outdir

    def test_sim_runs_headless_for_2s(self, temp_outdir):
        """Test that simulation runs in headless mode and produces valid CSV."""
        # Run simulation using subprocess to avoid Panda3D conflicts
        cmd = [
            sys.executable,
            "-m",
            "simulation.run_sim",
            "--headless",
            "--duration", "2.0",
            "--dt", "0.016",
            "--outdir", str(temp_outdir),
        ]

        # Run from repo root
        repo_root = Path(__file__).parent.parent.parent
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Check that it ran successfully
        assert result.returncode == 0, f"Simulation failed:\n{result.stderr}"

        # Find output CSV
        csv_files = list(temp_outdir.glob("run_*.csv"))
        assert len(csv_files) == 1, "Expected exactly one output CSV"

        csv_file = csv_files[0]

        # Read and validate CSV
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Should have data
        assert len(rows) > 0, "CSV should have data rows"

        # Check required columns
        expected_columns = [
            'time', 'pos_x', 'pos_y', 'pos_z',
            'vel_x', 'vel_y', 'vel_z',
            'accel_x', 'accel_y', 'accel_z',
            'path_s', 'segment_index'
        ]

        first_row = rows[0]
        for col in expected_columns:
            assert col in first_row, f"Missing column: {col}"

        # Check that time progresses
        times = [float(row['time']) for row in rows]
        assert times[0] >= 0
        assert times[-1] > times[0]
        assert times[-1] <= 2.1  # Should be close to 2s

        # Check that values are numeric and reasonable
        for row in rows:
            # All values should be parseable as floats
            for col in expected_columns:
                if col != 'segment_index':
                    val = float(row[col])
                    assert not np.isnan(val), f"NaN in {col}"
                    assert not np.isinf(val), f"Inf in {col}"

        print(f"\nTest passed: {len(rows)} frames logged over {times[-1]:.3f}s")

    def test_config_creation(self):
        """Test that SimConfig can be created with defaults."""
        config = SimConfig()
        assert config.duration > 0
        assert config.dt > 0
        assert config.speed > 0
        assert config.wakeboarder_mass > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
