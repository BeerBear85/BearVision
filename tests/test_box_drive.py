import os
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code' / 'modules'))

try:
    from modules.BoxHandler import BoxHandler
    from modules.ConfigurationHandler import ConfigurationHandler
except Exception as e:
    print("Exception occurred while importing modules:", e)
    BoxHandler = None

try:
    from boxsdk import Client  # noqa: F401
except Exception:
    Client = None


@pytest.mark.skip(reason="disabled to avoid network usage")
@pytest.mark.skipif(
    BoxHandler is None or Client is None,
    reason="Box dependencies missing or incompatible",
)
def test_box_upload_download(tmp_path):
    """
    Purpose:
        Upload and then download a file to verify round-trip behavior against
        Box. Network interactions are skipped in normal test runs.
    Inputs:
        tmp_path (Path): Temporary directory fixture provided by pytest.
    Outputs:
        None; assertions validate that the uploaded content matches the
        downloaded content.
    """
    if not os.getenv('STORAGE_CREDENTIALS_B64'):
        pytest.skip('STORAGE_CREDENTIALS_B64 not set')

    cfg_path = ROOT / 'config.ini'
    ConfigurationHandler.read_config_file(str(cfg_path))
    handler = BoxHandler()

    text = f"test-{uuid.uuid4().hex}"
    local_file = tmp_path / 'upload.txt'
    local_file.write_text(text)

    remote_path = f"test/{uuid.uuid4().hex}.txt"
    try:
        handler.upload_file(str(local_file), remote_path, overwrite=False)
        download_file = tmp_path / 'download.txt'
        handler.download_file(remote_path, str(download_file))
        assert download_file.read_text() == text
    finally:
        try:
            handler.delete_file(remote_path)
        except Exception:
            pass
