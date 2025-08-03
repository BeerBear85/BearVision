import base64
import sys
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / 'code' / 'modules'
sys.path.append(str(MODULE_DIR))

from tests.stubs.google_api import setup_google_modules, DummyCreds


def test_authenticate_uses_env(tmp_path, monkeypatch):
    captured = {}
    setup_google_modules(monkeypatch, captured)
    monkeypatch.chdir(tmp_path)

    import importlib
    module = importlib.import_module('GoogleDriveHandler')
    module = importlib.reload(module)
    GoogleDriveHandler = module.GoogleDriveHandler

    secret = base64.b64encode(b'{}').decode()
    monkeypatch.setenv('SECRET_ENV', secret)

    monkeypatch.setattr(
        module.service_account.Credentials,
        'from_service_account_info',
        lambda *a, **k: DummyCreds(),
        raising=False,
    )

    handler = GoogleDriveHandler({'GOOGLE_DRIVE': {'secret_key_name': 'SECRET_ENV'}})
    handler._authenticate()

    assert handler.service == 'service'
    assert isinstance(captured['creds'], DummyCreds)

