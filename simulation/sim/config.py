"""Configuration and CLI argument parsing for the simulation."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


# Cable tower locations in meters (local XY coordinates)
CABLE_TOWER_LOCATIONS: List[Tuple[float, float]] = [
    (0.00, 0.00),       # T0 reference
    (73.58, -55.61),    # T1
    (101.30, -50.12),   # T2
    (167.24, 80.92),    # T3
    (155.96, 126.04),   # T4
    (110.79, 114.50),   # T5
]


@dataclass
class SimConfig:
    """Simulation configuration."""

    headless: bool = False
    duration: float = 20.0  # seconds
    dt: float = 0.008  # physics timestep
    speed: float = 30.0 / 3.6  # m/s along cable (default: 30 km/h)
    outdir: Path = Path("simulation/out")
    seed: int = 0
    screenshot_every: float = 0.0  # seconds (0 = disabled)
    screenshot_dir: Path = None  # None = use outdir/screenshots

    # Physics parameters
    wakeboarder_mass: float = 80.0  # kg
    pull_force_gain: float = 5000.0  # N/m (proportional gain)
    pull_damping: float = 500.0  # N/(m/s) (derivative gain)
    z_target: float = 0.1  # target height above water
    z_spring: float = 1000.0  # vertical spring constant
    z_damping: float = 200.0  # vertical damping

    # Bullet physics
    bullet_substeps: int = 4
    bullet_solver_iterations: int = 10

    def __post_init__(self):
        """Convert paths to Path objects."""
        if isinstance(self.outdir, str):
            self.outdir = Path(self.outdir)
        if self.screenshot_dir is not None and isinstance(self.screenshot_dir, str):
            self.screenshot_dir = Path(self.screenshot_dir)


def parse_args() -> SimConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Wakeboard cable-park simulation"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no window)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Simulation duration in seconds (default: 20.0)"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.008,
        help="Physics timestep in seconds (default: 0.008)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=30.0 / 3.6,
        help="Wakeboarder speed along cable in m/s (default: 30 km/h = 8.33 m/s)"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("simulation/out"),
        help="Output directory (default: simulation/out)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)"
    )
    parser.add_argument(
        "--screenshot-every",
        type=float,
        default=0.0,
        help="Save screenshot every N seconds (0 = disabled)"
    )
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=None,
        help="Screenshot output directory (default: outdir/screenshots)"
    )

    args = parser.parse_args()

    return SimConfig(
        headless=args.headless,
        duration=args.duration,
        dt=args.dt,
        speed=args.speed,
        outdir=args.outdir,
        seed=args.seed,
        screenshot_every=args.screenshot_every,
        screenshot_dir=args.screenshot_dir,
    )
