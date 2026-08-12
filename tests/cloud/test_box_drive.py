"""Storage integration tests for the simulated and real Box implementations."""

from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bearvision.adapters import BoxStorageAdapter, SystemClock
from bearvision.config import load_edge_config
from bearvision.contracts import MediaAsset
from bearvision.integrations.box_handler import BoxHandler
from bearvision.ports import CapturedMedia
from bearvision.simulation import InMemoryStorage, VirtualClock


ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = b"BearVision Box storage integration test\n"


def _media() -> CapturedMedia:
    return CapturedMedia(
        asset=MediaAsset(
            asset_id=f"box-integration-{uuid.uuid4().hex}",
            filename="box-integration.txt",
            content_type="text/plain",
            size_bytes=len(PAYLOAD),
            created_at_utc=datetime.now(timezone.utc),
        ),
        content=PAYLOAD,
    )


async def _assert_storage_round_trip(storage, object_key: str) -> None:
    media = _media()
    uploaded = False
    try:
        first_receipt = await storage.upload(media, object_key)
        uploaded = True
        second_receipt = await storage.upload(media, object_key)

        assert second_receipt == first_receipt
        assert first_receipt.asset_id == media.asset.asset_id
        assert first_receipt.object_key == object_key
        assert first_receipt.checksum_sha256 == hashlib.sha256(PAYLOAD).hexdigest()
        assert await storage.download(object_key) == PAYLOAD
    finally:
        if uploaded:
            await storage.delete(object_key)


def test_simulated_box_storage_round_trip() -> None:
    """The cloud_storage=false substitute obeys the storage contract."""

    storage = InMemoryStorage(VirtualClock())
    object_key = f"box-integration-tests/{uuid.uuid4().hex}.txt"

    asyncio.run(_assert_storage_round_trip(storage, object_key))
    assert object_key not in storage.objects


@pytest.mark.box_integration
@pytest.mark.skipif(
    not os.getenv("STORAGE_CREDENTIALS_B64"),
    reason="STORAGE_CREDENTIALS_B64 is not set",
)
def test_real_box_storage_round_trip(tmp_path: Path) -> None:
    """The cloud_storage=true adapter performs a real Box API round trip."""

    config = load_edge_config(ROOT / "config" / "edge.yaml")
    handler_config = {
        "STORAGE_COMMON": {
            "secret_key_name": config.storage.credential_env,
            "secret_key_name_2": config.storage.secondary_credential_env or "",
        },
        "BOX": {"root_folder": config.storage.root_folder},
    }
    storage = BoxStorageAdapter(BoxHandler(handler_config), SystemClock(), tmp_path / "scratch")
    object_key = f"box-integration-tests/{uuid.uuid4().hex}.txt"

    failure_message = None
    try:
        asyncio.run(_assert_storage_round_trip(storage, object_key))
    except Exception as exc:
        # Keep credentials and signed JWT request bodies out of pytest tracebacks.
        failure_message = str(exc)
    if failure_message is not None:
        pytest.fail(f"Real Box integration failed: {failure_message}", pytrace=False)
