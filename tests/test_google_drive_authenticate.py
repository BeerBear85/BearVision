"""Tests for the GoogleDriveHandler authentication logic."""

import base64
import sys
import types
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / 'code' / 'modules'
sys.path.append(str(MODULE_DIR))

from tests.stubs.google_api import setup_google_modules, DummyCreds


def test_authenticate_service_account(tmp_path, monkeypatch):
    """Ensure service account credentials are used when configured."""
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

    handler = GoogleDriveHandler({'GOOGLE_DRIVE': {
        'secret_key_name': 'SECRET_ENV',
        'auth_mode': 'service',
    }})
    handler._authenticate()

    assert handler.service == 'service'
    assert isinstance(captured['creds'], DummyCreds)


def test_authenticate_user_flow(tmp_path, monkeypatch):
    """Verify OAuth user flow is used when auth_mode='user'."""
    captured = {}
    setup_google_modules(monkeypatch, captured)
    monkeypatch.chdir(tmp_path)

    import importlib
    module = importlib.import_module('GoogleDriveHandler')
    module = importlib.reload(module)
    GoogleDriveHandler = module.GoogleDriveHandler

    secret = base64.b64encode(b'{}').decode()
    monkeypatch.setenv('SECRET_ENV', secret)

    class DummyFlow:
        """Minimal stand-in for InstalledAppFlow."""

        def run_local_server(self, port=0):  # pragma: no cover - simple stub
            return DummyCreds()

    def fake_from_client_config(*a, **k):  # pragma: no cover - simple stub
        return DummyFlow()

    monkeypatch.setattr(module, 'InstalledAppFlow', types.SimpleNamespace(
        from_client_config=fake_from_client_config
    ))

    handler = GoogleDriveHandler({'GOOGLE_DRIVE': {
        'secret_key_name': 'SECRET_ENV',
        'auth_mode': 'user',
    }})
    handler._authenticate()

    assert handler.service == 'service'
    assert isinstance(captured['creds'], DummyCreds)

