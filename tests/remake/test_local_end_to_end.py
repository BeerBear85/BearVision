import asyncio
from datetime import timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from bearvision.contracts import load_scenario
from bearvision.edge import build_behavioral_system
from bearvision.server import (
    BearTagAssignment,
    BearTagRecord,
    FileSystemJobQueue,
    FileUserRegistry,
    RegistryData,
    ServerWorker,
    UserRecord,
)
from bearvision.server.admin import AdminCatalog
from bearvision.simulation import VirtualClock


ROOT = Path(__file__).resolve().parents[2]


def test_simulated_edge_shared_folder_server_worker_and_admin_view(tmp_path: Path) -> None:
    queue_root = tmp_path / "shared-queue"
    edge_queue = FileSystemJobQueue(queue_root)
    scenario = load_scenario(ROOT / "specs/scenarios/single-rider-success.yaml")

    edge_result = build_behavioral_system(
        scenario,
        job_queue=edge_queue,
        process_server=False,
    ).run()

    assert edge_result.failures == ()
    assert edge_result.expectation_failures == ()
    assert edge_result.assignments == ()
    assert edge_queue.snapshot()["counts"]["ready"] == 1

    clock = VirtualClock()
    email = "rider-17@scenario.invalid"
    user_id = uuid5(NAMESPACE_URL, f"bearvision:scenario-user:{email}")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        RegistryData(
            users=(UserRecord(id=user_id, email=email, displayName="Rider 17"),),
            bearTags=(BearTagRecord(id="tag-17"),),
            assignments=(
                BearTagAssignment(
                    id="assignment-tag-17",
                    userId=user_id,
                    bearTagId="tag-17",
                    validFrom=clock.start_utc - timedelta(days=1),
                    validTo=clock.start_utc + timedelta(days=1),
                ),
            ),
        ).model_dump_json(by_alias=True, indent=2),
        encoding="utf-8",
    )
    registry = FileUserRegistry(registry_path)

    server_result = asyncio.run(
        ServerWorker(FileSystemJobQueue(queue_root), registry, clock).run_once()
    )

    assert server_result is not None and server_result.status == "processed"
    assert server_result.selected_user_id == user_id
    assert (queue_root / "processed" / f"user_{user_id}" / server_result.job_id).is_dir()
    jobs = asyncio.run(
        AdminCatalog(FileSystemJobQueue(queue_root), registry).list_jobs(
            status="processed"
        )
    )
    assert jobs["items"][0]["displayName"] == "Rider 17"
    assert jobs["items"][0]["userEmail"] == email
