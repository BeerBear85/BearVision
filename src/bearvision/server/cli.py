"""CLI process boundary used by the local Node.js control service."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys

from bearvision.adapters import BoxJobQueue, SystemClock
from bearvision.config import ServerConfig, load_server_config
from bearvision.integrations.box_handler import BoxHandler
from bearvision.ports import ComponentTimeout, ComponentUnavailable, ManagedJobQueue

from .commands import (
    ServerCommandModule,
    parse_command,
    serialize_result,
    write_worker_status,
)
from .queue import FileSystemJobQueue
from .registry import FileUserRegistry
from .worker import ServerWorker


logger = logging.getLogger(__name__)


def _resolve(base: Path, value: Path) -> Path:
    return value if value.is_absolute() else base / value


def build_runtime(
    config_path: Path,
) -> tuple[ServerConfig, ManagedJobQueue, FileUserRegistry]:
    config = load_server_config(config_path)
    base = config_path.resolve().parents[1]
    registry = FileUserRegistry(_resolve(base, config.registry_path))
    if config.local_queue_root is not None:
        queue: ManagedJobQueue = FileSystemJobQueue(
            _resolve(base, config.local_queue_root)
        )
    else:
        box_config = {
            "STORAGE_COMMON": {
                "secret_key_name": config.storage.credential_env,
                "secret_key_name_2": config.storage.secondary_credential_env or "",
            },
            "BOX": {"root_folder": config.storage.root_folder},
        }
        queue = BoxJobQueue(
            BoxHandler(box_config),
            _resolve(base, config.scratch_dir),
        )
    return config, queue, registry


async def worker_loop(
    config_path: Path,
    config: ServerConfig,
    queue: ManagedJobQueue,
    registry: FileUserRegistry,
) -> None:
    worker = ServerWorker(queue, registry, SystemClock(), config.assignment)
    write_worker_status(config_path, status="idle")
    while True:
        try:
            result = await worker.run_once()
            if result is None:
                write_worker_status(config_path, status="idle")
                await asyncio.sleep(config.worker.poll_interval_s)
            else:
                write_worker_status(
                    config_path,
                    status="idle",
                    lastJobId=result.job_id,
                    lastResult=result.status,
                )
        except (ComponentTimeout, ComponentUnavailable) as exc:
            write_worker_status(config_path, status="retrying", error=str(exc))
            await asyncio.sleep(config.worker.retry_delay_s)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="BearVision server worker and admin bridge")
    result.add_argument("--config", type=Path, default=Path("config/server.yaml"))
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("execute")
    commands.add_parser("worker")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config, queue, registry = build_runtime(args.config)
        if args.command == "worker":
            asyncio.run(worker_loop(args.config, config, queue, registry))
            return 0
        payload = sys.stdin.read(65_537)
        if len(payload) > 65_536:
            raise ValueError("command envelope is too large")
        command = parse_command(payload)
        output = asyncio.run(
            ServerCommandModule(args.config, config, queue, registry).execute(command)
        )
        print(serialize_result(output))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
