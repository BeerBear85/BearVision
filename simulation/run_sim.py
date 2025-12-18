#!/usr/bin/env python
"""Entry point for wakeboard cable-park simulation.

Usage:
    python -m simulation.run_sim --duration 30 --speed 10
    python -m simulation.run_sim --headless --duration 10
"""

from simulation.sim.app import run_simulation
from simulation.sim.config import parse_args


def main():
    """Main entry point."""
    config = parse_args()

    print("=" * 60)
    print("Wakeboard Cable-Park Simulation")
    print("=" * 60)
    print(f"Mode:     {'Headless' if config.headless else 'Visual'}")
    print(f"Duration: {config.duration}s")
    print(f"Timestep: {config.dt}s")
    print(f"Speed:    {config.speed} m/s")
    print(f"Output:   {config.outdir}")
    print("=" * 60)
    print()

    run_simulation(config)


if __name__ == "__main__":
    main()
