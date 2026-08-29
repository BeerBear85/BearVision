"""Process boundary used by the thin Node.js Edge control server."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from bearvision.config import load_edge_config
from bearvision.contracts import RuntimeEventKind, serialize_runtime_event
from bearvision.edge import build_real_orchestrator
from bearvision.simulation import ReplayOptions, ScenarioExecution


def emit(
    kind: RuntimeEventKind,
    payload: dict | None = None,
    *,
    at_s: float | None = None,
) -> None:
    print(serialize_runtime_event(kind, payload or {}, at_s=at_s), flush=True)


def simulate(
    path: Path,
    *,
    realtime: bool,
    speed: float,
    local_queue_root: Path | None = None,
    config_path: Path = Path("config/edge.yaml"),
) -> int:
    replay = ReplayOptions(
        realtime=realtime,
        speed=speed,
        include_server_assignments=False,
    )
    execution = ScenarioExecution.run(
        path,
        config_path=config_path,
        local_queue_root=local_queue_root,
    )
    for event in execution.replay(replay):
        emit(event.kind, event.payload, at_s=event.at_s)
    return execution.exit_code


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
