"""Process boundary used by the thin Node.js Edge control server."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from bearvision.config import load_edge_config
from bearvision.contracts import RuntimeEventKind, serialize_runtime_event
from bearvision.edge import build_real_orchestrator
from bearvision.edge.preflight import check_edge_readiness
from bearvision.simulation import ReplayOptions, ScenarioExecution


def emit(
    kind: RuntimeEventKind,
    payload: dict | None = None,
    *,
    run_id: str,
    at_s: float | None = None,
) -> None:
    print(
        serialize_runtime_event(
            kind,
            payload or {},
            run_id=run_id,
            emitted_at=datetime.now(timezone.utc),
            at_s=at_s,
        ),
        flush=True,
    )


def simulate(
    path: Path,
    *,
    run_id: str,
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
        emit(event.kind, event.payload, run_id=run_id, at_s=event.at_s)
    return execution.exit_code


def readiness(
    config_path: Path,
    *,
    capture_dir: Path,
    scratch_dir: Path,
) -> dict[str, object]:
    """Return the Python-owned hardware readiness report used by Edge Control."""

    return check_edge_readiness(
        config_path,
        capture_dir=capture_dir,
        scratch_dir=scratch_dir,
    ).to_dict()


async def hardware(
    config_path: Path,
    *,
    run_id: str,
    capture_dir: Path = Path("temp/captures"),
    scratch_dir: Path = Path("temp/scratch"),
) -> int:
    config = load_edge_config(config_path)
    logging.basicConfig(
        level=getattr(logging, config.system.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    orchestrator = build_real_orchestrator(
        config,
        capture_dir=capture_dir,
        scratch_dir=scratch_dir,
        event_sink=lambda event: emit(event.kind, event.payload, run_id=run_id),
    )
    emit("hardware_initializing", {"config": str(config_path)}, run_id=run_id)
    command_task = asyncio.create_task(_read_control_commands(orchestrator))
    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        emit("hardware_stopping", {"reason": "interrupt"}, run_id=run_id)
    except Exception as exc:
        emit(
            "component_failed",
            {
                "component": "runtime",
                "error": str(exc),
                "stage": "failed",
                "operator_message": "The Edge runtime stopped because an operation failed.",
                "corrective_action": "Review the failed stage and restart the runtime.",
                "severity": "terminal",
                "retryable": False,
            },
            run_id=run_id,
        )
        return 1
    finally:
        command_task.cancel()
    return 0


async def process_control_command(orchestrator: Any, command: dict[str, object]) -> None:
    """Apply one versioned command received from the Node supervisor."""

    if command.get("command_version") != "1.0":
        raise ValueError("unsupported control command version")
    if command.get("kind") != "retry_failure":
        raise ValueError("unsupported control command kind")
    failure_id = command.get("failure_id")
    if not isinstance(failure_id, str) or not failure_id:
        raise ValueError("retry_failure requires a failure_id")
    await orchestrator.retry_failure(failure_id)


async def _read_control_commands(orchestrator: Any) -> None:
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            return
        try:
            command = json.loads(line)
            if not isinstance(command, dict):
                raise ValueError("control command must be an object")
            await process_control_command(orchestrator, command)
        except Exception as exc:
            logging.getLogger(__name__).error("Control command failed: %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="BearVision Edge control process")
    commands = parser.add_subparsers(dest="command", required=True)
    simulation = commands.add_parser("simulate")
    simulation.add_argument("scenario", type=Path)
    simulation.add_argument("--realtime", action="store_true")
    simulation.add_argument("--speed", type=float, default=1.0)
    simulation.add_argument("--local-queue-root", type=Path)
    simulation.add_argument("--config", type=Path, default=Path("config/edge.yaml"))
    simulation.add_argument("--run-id", default=f"local-{uuid4()}")
    real = commands.add_parser("hardware")
    real.add_argument("--config", type=Path, default=Path("config/edge.yaml"))
    real.add_argument("--capture-dir", type=Path, default=Path("temp/captures"))
    real.add_argument("--scratch-dir", type=Path, default=Path("temp/scratch"))
    real.add_argument("--run-id", default=f"local-{uuid4()}")
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--config", type=Path, default=Path("config/edge.yaml"))
    preflight.add_argument("--capture-dir", type=Path, default=Path("temp/captures"))
    preflight.add_argument("--scratch-dir", type=Path, default=Path("temp/scratch"))
    args = parser.parse_args()
    if args.command == "simulate":
        return simulate(
            args.scenario,
            run_id=args.run_id,
            realtime=args.realtime,
            speed=args.speed,
            local_queue_root=args.local_queue_root,
            config_path=args.config,
        )
    if args.command == "preflight":
        report = readiness(
            args.config,
            capture_dir=args.capture_dir,
            scratch_dir=args.scratch_dir,
        )
        print(json.dumps(report), flush=True)
        return 1 if report["blocking"] else 0
    return asyncio.run(
        hardware(
            args.config,
            run_id=args.run_id,
            capture_dir=args.capture_dir,
            scratch_dir=args.scratch_dir,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
