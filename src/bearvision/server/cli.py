"""CLI process boundary used by the local Node.js control service."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from bearvision.adapters import BoxJobQueue, SystemClock
from bearvision.config import ServerConfig, load_server_config
from bearvision.integrations.box_handler import BoxHandler
from bearvision.ports import ComponentTimeout, ComponentUnavailable, ManagedJobQueue

from .admin import AdminCatalog, AdminMediaService
from .queue import FileSystemJobQueue
from .registry import BearTagAssignment, FileUserRegistry
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


def _status_path(config_path: Path) -> Path:
    return config_path.resolve().parents[1] / "temp/server-worker-status.json"


def _write_status(config_path: Path, **values: Any) -> None:
    path = _status_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    payload = {
        "pid": os.getpid(),
        "updatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **values,
    }
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def snapshot(
    config_path: Path, queue: ManagedJobQueue, registry: FileUserRegistry
) -> dict:
    worker = _worker_status(config_path)
    return {
        "worker": worker,
        "queue": queue.snapshot(),
        "registry": registry.load().model_dump(mode="json", by_alias=True),
    }


def _worker_status(config_path: Path) -> dict[str, Any]:
    status_path = _status_path(config_path)
    worker: dict[str, Any] = {"status": "stopped"}
    if status_path.exists():
        try:
            worker = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            worker = {"status": "unknown"}
    return worker


async def worker_loop(
    config_path: Path,
    config: ServerConfig,
    queue: ManagedJobQueue,
    registry: FileUserRegistry,
) -> None:
    worker = ServerWorker(queue, registry, SystemClock(), config.assignment)
    _write_status(config_path, status="idle")
    while True:
        try:
            result = await worker.run_once()
            if result is None:
                _write_status(config_path, status="idle")
                await asyncio.sleep(config.worker.poll_interval_s)
            else:
                _write_status(
                    config_path,
                    status="idle",
                    lastJobId=result.job_id,
                    lastResult=result.status,
                )
        except (ComponentTimeout, ComponentUnavailable) as exc:
            _write_status(config_path, status="retrying", error=str(exc))
            await asyncio.sleep(config.worker.retry_delay_s)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="BearVision server worker and admin bridge")
    result.add_argument("--config", type=Path, default=Path("config/server.yaml"))
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("snapshot")
    commands.add_parser("summary")
    commands.add_parser("run-once")
    commands.add_parser("worker")
    list_jobs = commands.add_parser("list-jobs")
    list_jobs.add_argument("--page", type=int, default=1)
    list_jobs.add_argument("--page-size", type=int, default=24)
    list_jobs.add_argument("--status")
    list_jobs.add_argument("--query", default="")
    list_jobs.add_argument("--user-id")
    job_detail = commands.add_parser("job-detail")
    job_detail.add_argument("--job-id", required=True)
    list_users = commands.add_parser("list-users")
    list_users.add_argument("--page", type=int, default=1)
    list_users.add_argument("--page-size", type=int, default=50)
    list_users.add_argument("--query", default="")
    commands.add_parser("list-tags")
    media = commands.add_parser("materialize-media")
    media.add_argument("--job-id", required=True)
    media.add_argument("--kind", choices=("video", "thumbnail"), required=True)
    create_user = commands.add_parser("create-user")
    create_user.add_argument("--email", required=True)
    create_user.add_argument("--display-name", required=True)
    create_tag = commands.add_parser("create-tag")
    create_tag.add_argument("--id", required=True)
    create_assignment = commands.add_parser("create-assignment")
    create_assignment.add_argument("--id")
    create_assignment.add_argument("--user-id", required=True)
    create_assignment.add_argument("--bear-tag-id", required=True)
    create_assignment.add_argument("--valid-from", required=True)
    create_assignment.add_argument("--valid-to", required=True)
    validate_assignment = commands.add_parser("validate-assignment")
    validate_assignment.add_argument("--id")
    validate_assignment.add_argument("--user-id", required=True)
    validate_assignment.add_argument("--bear-tag-id", required=True)
    validate_assignment.add_argument("--valid-from", required=True)
    validate_assignment.add_argument("--valid-to", required=True)
    requeue = commands.add_parser("requeue")
    requeue.add_argument("--job-id", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config, queue, registry = build_runtime(args.config)
        catalog = AdminCatalog(queue, registry)
        output: object
        if args.command == "snapshot":
            output = snapshot(args.config, queue, registry)
        elif args.command == "summary":
            output = asyncio.run(catalog.summary())
            output["worker"] = _worker_status(args.config)
        elif args.command == "list-jobs":
            output = asyncio.run(
                catalog.list_jobs(
                    page=args.page,
                    page_size=args.page_size,
                    status=args.status,
                    query=args.query,
                    user_id=args.user_id,
                )
            )
        elif args.command == "job-detail":
            output = asyncio.run(catalog.get_job(args.job_id))
        elif args.command == "list-users":
            output = asyncio.run(
                catalog.list_users(
                    page=args.page, page_size=args.page_size, query=args.query
                )
            )
        elif args.command == "list-tags":
            output = catalog.list_bear_tags()
        elif args.command == "materialize-media":
            base = args.config.resolve().parents[1]
            media = AdminMediaService(
                queue, _resolve(base, config.scratch_dir) / "admin-media"
            )
            output = asyncio.run(media.materialize(args.job_id, args.kind))
        elif args.command == "create-user":
            output = registry.create_user(args.email, args.display_name).model_dump(
                mode="json", by_alias=True
            )
        elif args.command == "create-tag":
            output = registry.create_bear_tag(args.id).model_dump(mode="json", by_alias=True)
        elif args.command == "create-assignment":
            output = registry.create_assignment(
                BearTagAssignment(
                    id=args.id or f"assignment-{uuid4().hex}",
                    userId=args.user_id,
                    bearTagId=args.bear_tag_id,
                    validFrom=args.valid_from,
                    validTo=args.valid_to,
                )
            ).model_dump(mode="json", by_alias=True)
        elif args.command == "validate-assignment":
            assignment, _ = registry.validate_assignment(
                BearTagAssignment(
                    id=args.id or f"assignment-{uuid4().hex}",
                    userId=args.user_id,
                    bearTagId=args.bear_tag_id,
                    validFrom=args.valid_from,
                    validTo=args.valid_to,
                )
            )
            output = {
                "valid": True,
                "assignment": assignment.model_dump(mode="json", by_alias=True),
            }
        elif args.command == "requeue":
            output = {"requeued": asyncio.run(queue.requeue(args.job_id))}
        elif args.command == "run-once":
            result = asyncio.run(
                ServerWorker(queue, registry, SystemClock(), config.assignment).run_once()
            )
            output = result.model_dump(mode="json", by_alias=True) if result else None
        else:
            asyncio.run(worker_loop(args.config, config, queue, registry))
            return 0
        print(json.dumps(output, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
