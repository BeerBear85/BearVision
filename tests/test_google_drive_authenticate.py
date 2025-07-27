import json
import os
import sys
import types
import base64
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / 'code' / 'modules'
sys.path.append(str(MODULE_DIR))


class DummyCreds:
    pass


def setup_google_modules(monkeypatch, captured_info):
    import google.oauth2.service_account as sa_mod
    import googleapiclient.discovery as disc_mod
    import googleapiclient.http as http_mod

    class DummyMediaFileUpload:
        def __init__(self, filename, resumable=False):
            self.filename = filename

    class DummyMediaIoBaseDownload:
        def __init__(self, fh, request):
            self.fh = fh
            self.request = request
            self.done = False

        def next_chunk(self):
            if not self.done:
                self.fh.write(self.request.content)
                self.done = True
            return None, self.done

    def fake_from_info(info, scopes=None):
        captured_info['info'] = info
        return DummyCreds()

    monkeypatch.setattr(sa_mod.Credentials, 'from_service_account_info', fake_from_info, raising=False)
    monkeypatch.setattr(disc_mod, 'build', lambda *a, **k: 'service', raising=False)
    monkeypatch.setattr(http_mod, 'MediaFileUpload', DummyMediaFileUpload, raising=False)
    monkeypatch.setattr(http_mod, 'MediaIoBaseDownload', DummyMediaIoBaseDownload, raising=False)


def test_authenticate_accepts_file_path(tmp_path, monkeypatch):
    captured = {}
    setup_google_modules(monkeypatch, captured)

    import importlib
    module = importlib.import_module('GoogleDriveHandler')
    module = importlib.reload(module)
    GoogleDriveHandler = module.GoogleDriveHandler

    creds = {'a': 1}
    path = tmp_path / 'creds.json'
    encoded = base64.b64encode(json.dumps(creds).encode()).decode()
    path.write_text(encoded)

    monkeypatch.setenv('CREDS_ENV', str(path))
    handler = GoogleDriveHandler({'GOOGLE_DRIVE': {'secret_key_name': 'CREDS_ENV'}})
    handler._authenticate()
    assert handler.service == 'service'


def test_authenticate_accepts_base64_value(monkeypatch):
    captured = {}
    setup_google_modules(monkeypatch, captured)

    import importlib
    module = importlib.import_module('GoogleDriveHandler')
    module = importlib.reload(module)
    GoogleDriveHandler = module.GoogleDriveHandler

    creds = {'b': 2}
    encoded = base64.b64encode(json.dumps(creds).encode()).decode()

    monkeypatch.setenv('CREDS_ENV', encoded)
    handler = GoogleDriveHandler({'GOOGLE_DRIVE': {'secret_key_name': 'CREDS_ENV'}})
    handler._authenticate()
    assert handler.service == 'service'

