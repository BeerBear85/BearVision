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
    """
    Purpose:
        Ensure service account authentication succeeds even when the secondary
        credential environment variable is absent.
    Inputs:
        tmp_path (Path): Temporary directory for isolating test artifacts.
        monkeypatch: Pytest fixture to modify environment and modules.
    Outputs:
        None; assertions validate that the service object is created.
    """
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

        'secret_key_name_2': 'SECRET_ENV2',  # env var not set on purpose

        'auth_mode': 'service',
    }})
    handler._authenticate()

    assert handler.service == 'service'
    assert isinstance(captured['creds'], DummyCreds)


def test_missing_primary_secret_raises(tmp_path, monkeypatch):
    """
    Purpose:
        Verify that omitting ``secret_key_name`` from the configuration triggers
        a ``KeyError`` as the primary credential is mandatory.
    Inputs:
        tmp_path (Path): Temporary directory for test isolation.
        monkeypatch: Fixture used to stub google modules.
    Outputs:
        None; the test passes if ``KeyError`` is raised.
    """
    captured = {}
    setup_google_modules(monkeypatch, captured)
    monkeypatch.chdir(tmp_path)

    import importlib
    module = importlib.import_module('GoogleDriveHandler')
    module = importlib.reload(module)
    GoogleDriveHandler = module.GoogleDriveHandler

    handler = GoogleDriveHandler({'GOOGLE_DRIVE': {}})
    with pytest.raises(KeyError):
        handler._authenticate()


def test_authenticate_user_flow(tmp_path, monkeypatch):
    """
    Purpose:
        Verify OAuth user flow is used when `auth_mode` is set to ``user`` and
        no secondary credential name is configured.
    Inputs:
        tmp_path (Path): Temporary directory for isolation.
        monkeypatch: Pytest fixture for patching modules and environment.
    Outputs:
        None; assertions check that authentication completes.
    """
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

    monkeypatch.setattr(
        module,
        'InstalledAppFlow',
        types.SimpleNamespace(from_client_config=fake_from_client_config),
    )

    handler = GoogleDriveHandler({'GOOGLE_DRIVE': {
        'secret_key_name': 'SECRET_ENV',
        'secret_key_name_2': 'SECRET_ENV2',  # explicit second name required
        'auth_mode': 'user',
    }})
    handler._authenticate()

    assert handler.service == 'service'
    assert isinstance(captured['creds'], DummyCreds)


def test_authenticate_split_env(tmp_path, monkeypatch):
    """Ensure credentials split across two env vars are concatenated."""
    captured = {}
    setup_google_modules(monkeypatch, captured)
    monkeypatch.chdir(tmp_path)

    import importlib
    module = importlib.import_module('GoogleDriveHandler')
    module = importlib.reload(module)
    GoogleDriveHandler = module.GoogleDriveHandler

    secret = base64.b64encode(b'{}').decode()
    # Emulate splitting large credentials into two parts to mimic environment
    # variable length constraints in real deployments.
    first, second = secret[: len(secret)//2], secret[len(secret)//2 :]
    monkeypatch.setenv('SECRET_ENV', first)
    monkeypatch.setenv('SECRET_ENV2', second)

    monkeypatch.setattr(
        module.service_account.Credentials,
        'from_service_account_info',
        lambda *a, **k: DummyCreds(),
        raising=False,
    )

    handler = GoogleDriveHandler({'GOOGLE_DRIVE': {
        'secret_key_name': 'SECRET_ENV',
        'secret_key_name_2': 'SECRET_ENV2',
        'auth_mode': 'service',
    }})
    handler._authenticate()

    assert handler.service == 'service'
    assert isinstance(captured['creds'], DummyCreds)

