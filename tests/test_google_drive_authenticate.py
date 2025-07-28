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


# ---------------------------------------------------------------------------
# Helpers to ensure the tests run even when the real Google packages are
# missing or incomplete. Some versions of ``google-api-python-client`` try to
# import an optional ``discovery_cache`` module on import.  If that import fails
# the whole ``googleapiclient.discovery`` module cannot be loaded.  To make the
# tests robust we provide lightweight stub modules whenever the real ones are
# unavailable.
# ---------------------------------------------------------------------------
sys.modules.setdefault('google', types.ModuleType('google'))

try:
    import googleapiclient.discovery  # noqa: F401
    import googleapiclient.http  # noqa: F401
    import google.oauth2.service_account  # noqa: F401
except Exception:  # pragma: no cover - only executed when deps are missing
    oauth2_mod = types.ModuleType('google.oauth2')
    service_account_mod = types.ModuleType('google.oauth2.service_account')

    service_account_mod.Credentials = types.SimpleNamespace(
        from_service_account_info=lambda info, scopes=None: DummyCreds()
    )
    oauth2_mod.service_account = service_account_mod

    apiclient_mod = types.ModuleType('googleapiclient')
    disc_mod = types.ModuleType('googleapiclient.discovery')
    http_mod = types.ModuleType('googleapiclient.http')

    disc_mod.build = lambda *a, **k: None

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

    http_mod.MediaFileUpload = DummyMediaFileUpload
    http_mod.MediaIoBaseDownload = DummyMediaIoBaseDownload

    sys.modules.setdefault('google.oauth2', oauth2_mod)
    sys.modules.setdefault('google.oauth2.service_account', service_account_mod)
    sys.modules.setdefault('googleapiclient', apiclient_mod)
    sys.modules.setdefault('googleapiclient.discovery', disc_mod)
    sys.modules.setdefault('googleapiclient.http', http_mod)


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

