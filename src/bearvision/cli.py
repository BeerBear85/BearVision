"""Command-line entrypoints for BearVision 3."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from bearvision.config import load_edge_config
from bearvision.contracts import serialize_runtime_event
from bearvision.edge import build_real_orchestrator
from bearvision.simulation import ReplayOptions, ScenarioExecution


def edge_main() -> int:
    parser = argparse.ArgumentParser(description="Run the BearVision 3 edge service")
    parser.add_argument("--config", type=Path, default=Path("config/edge.yaml"))
    parser.add_argument("--capture-dir", type=Path, default=Path("temp/captures"))
    parser.add_argument("--scratch-dir", type=Path, default=Path("temp/scratch"))
    args = parser.parse_args()
    config = load_edge_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config.system.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    orchestrator = build_real_orchestrator(
        config,
        capture_dir=args.capture_dir,
        scratch_dir=args.scratch_dir,
    )
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("BearVision shutdown requested")
    return 0


def simulate_main() -> int:
    parser = argparse.ArgumentParser(description="Run a BearVision behavioural scenario")
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/edge.yaml"))
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Replay the deterministic trace at wall-clock speed for monitoring clients",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed used with --realtime (1.0 is wall-clock speed)",
    )
    args = parser.parse_args()
    try:
        replay = ReplayOptions(
            realtime=args.realtime,
            speed=args.speed,
            include_server_assignments=True,
        )
    except ValueError as exc:
        parser.error(str(exc))
    execution = ScenarioExecution.run(args.scenario, config_path=args.config)
    for event in execution.replay(replay):
        print(
            serialize_runtime_event(event.kind, event.payload, at_s=event.at_s),
            flush=True,
        )
    return execution.exit_code
