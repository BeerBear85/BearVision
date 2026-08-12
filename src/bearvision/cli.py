"""Command-line entrypoints for BearVision 3."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from bearvision.config import load_edge_config
from bearvision.contracts import load_scenario
from bearvision.edge import build_behavioral_system, build_real_orchestrator


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
    args = parser.parse_args()
    result = build_behavioral_system(load_scenario(args.scenario)).run()
    for entry in result.trace:
        print(
            json.dumps(
                {"at_s": entry.at_s, "kind": entry.kind, "payload": entry.payload},
                default=str,
            )
        )
    for failure in result.expectation_failures:
        logging.error("Expectation failed: %s", failure)
    return 1 if result.failures or result.expectation_failures else 0
