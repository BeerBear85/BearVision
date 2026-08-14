"""Process boundary used by the thin Node.js Edge control server."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
import time

from bearvision.config import load_edge_config
from bearvision.contracts import load_scenario
from bearvision.edge import build_behavioral_system, build_real_orchestrator
from bearvision.server import FileSystemJobQueue


def emit(kind: str, payload: dict | None = None, *, at_s: float | None = None) -> None:
    print(
        json.dumps(
            {
                "control_event_version": "1.0",
                "at_s": at_s,
                "kind": kind,
                "payload": payload or {},
            },
            default=str,
        ),
        flush=True,
    )


def simulate(
    path: Path,
    *,
    realtime: bool,
    speed: float,
    local_queue_root: Path | None = None,
    config_path: Path = Path("config/edge.yaml"),
) -> int:
    if speed <= 0:
        raise ValueError("speed must be positive")
    queue = FileSystemJobQueue(local_queue_root) if local_queue_root else None
    result = build_behavioral_system(
        load_scenario(path),
        edge_config=load_edge_config(config_path),
        job_queue=queue,
        process_server=queue is None,
    ).run()
    previous_at_s = 0.0
    for entry in result.trace:
        if entry.kind == "server_assignment":
            continue
        if realtime:
            time.sleep(max(0.0, entry.at_s - previous_at_s) / speed)
        previous_at_s = entry.at_s
        emit(entry.kind, entry.payload, at_s=entry.at_s)
    for expectation_failure in result.expectation_failures:
        emit("expectation_failed", {"message": expectation_failure})
    for component_failure in result.failures:
        emit("component_failed", component_failure)
    return 1 if result.failures or result.expectation_failures else 0


async def hardware(config_path: Path) -> int:
    config = load_edge_config(config_path)
    logging.basicConfig(
        level=getattr(logging, config.system.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    orchestrator = build_real_orchestrator(
        config,
        capture_dir=Path("temp/captures"),
        scratch_dir=Path("temp/scratch"),
    )
    emit("hardware_initializing", {"config": str(config_path)})
    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        emit("hardware_stopping", {"reason": "interrupt"})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="BearVision Edge control process")
    commands = parser.add_subparsers(dest="command", required=True)
    simulation = commands.add_parser("simulate")
    simulation.add_argument("scenario", type=Path)
    simulation.add_argument("--realtime", action="store_true")
    simulation.add_argument("--speed", type=float, default=1.0)
    simulation.add_argument("--local-queue-root", type=Path)
    simulation.add_argument("--config", type=Path, default=Path("config/edge.yaml"))
    real = commands.add_parser("hardware")
    real.add_argument("--config", type=Path, default=Path("config/edge.yaml"))
    args = parser.parse_args()
    if args.command == "simulate":
        return simulate(
            args.scenario,
            realtime=args.realtime,
            speed=args.speed,
            local_queue_root=args.local_queue_root,
            config_path=args.config,
        )
    return asyncio.run(hardware(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
