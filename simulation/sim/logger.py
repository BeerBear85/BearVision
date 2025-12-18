"""CSV logging for simulation data."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO

import numpy as np


class SimulationLogger:
    """Logs simulation data to CSV file."""

    def __init__(self, output_dir: Path):
        """Initialize logger.

        Args:
            output_dir: Directory to save log files
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = self.output_dir / f"run_{timestamp}.csv"

        self.file: Optional[TextIO] = None
        self.writer: Optional[csv.DictWriter] = None

        # Track previous velocity for acceleration calculation
        self.prev_velocity: Optional[np.ndarray] = None
        self.dt: float = 0.008

    def start(self, dt: float) -> None:
        """Start logging, open file and write header.

        Args:
            dt: Simulation timestep for acceleration calculation
        """
        self.dt = dt
        self.file = open(self.filepath, 'w', newline='')

        fieldnames = [
            'time',
            'pos_x', 'pos_y', 'pos_z',
            'vel_x', 'vel_y', 'vel_z',
            'accel_x', 'accel_y', 'accel_z',
            'path_s',
            'segment_index',
        ]

        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.prev_velocity = None

    def log_frame(
        self,
        time: float,
        position: np.ndarray,
        velocity: np.ndarray,
        path_s: float,
        segment_index: int,
    ) -> None:
        """Log a single frame of data.

        Args:
            time: Simulation time in seconds
            position: Position [x, y, z] in meters
            velocity: Velocity [vx, vy, vz] in m/s
            path_s: Distance along cable path in meters
            segment_index: Current cable segment index
        """
        if self.writer is None:
            raise RuntimeError("Logger not started. Call start() first.")

        # Calculate acceleration from finite difference
        if self.prev_velocity is not None:
            accel = (velocity - self.prev_velocity) / self.dt
        else:
            accel = np.zeros(3)

        self.prev_velocity = velocity.copy()

        # Write row
        row = {
            'time': f'{time:.6f}',
            'pos_x': f'{position[0]:.6f}',
            'pos_y': f'{position[1]:.6f}',
            'pos_z': f'{position[2]:.6f}',
            'vel_x': f'{velocity[0]:.6f}',
            'vel_y': f'{velocity[1]:.6f}',
            'vel_z': f'{velocity[2]:.6f}',
            'accel_x': f'{accel[0]:.6f}',
            'accel_y': f'{accel[1]:.6f}',
            'accel_z': f'{accel[2]:.6f}',
            'path_s': f'{path_s:.6f}',
            'segment_index': str(segment_index),
        }

        self.writer.writerow(row)

    def stop(self) -> Path:
        """Stop logging and close file.

        Returns:
            Path to the output file
        """
        if self.file is not None:
            self.file.close()
            self.file = None
            self.writer = None

        return self.filepath
