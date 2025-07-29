import os
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code' / 'modules'))


try:
    from modules.GoogleDriveHandler import GoogleDriveHandler
    from modules.ConfigurationHandler import ConfigurationHandler
except Exception as e:
    print("Exception occurred while importing modules:", e)
    GoogleDriveHandler = None

try:
    from googleapiclient.http import build_http  # noqa: F401
except Exception:
    build_http = None


@pytest.mark.skipif(
    GoogleDriveHandler is None or build_http is None,
    reason="Google Drive dependencies missing or incompatible",
)
def test_google_drive_upload_download(tmp_path):
    if not os.getenv('GOOGLE_CREDENTIALS_JSON'):
        pytest.skip('GOOGLE_CREDENTIALS_JSON not set')

    cfg_path = ROOT / 'config.ini'
    ConfigurationHandler.read_config_file(str(cfg_path))
    handler = GoogleDriveHandler()

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
