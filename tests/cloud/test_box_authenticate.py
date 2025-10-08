"""
Tests for BoxHandler authentication logic.

This test file focuses specifically on authentication mechanisms including
service account authentication, split environment variables, and error
handling for missing credentials.

For core functionality testing with mocks, see test_box_handler.py
For actual Box API integration testing, see test_box_drive.py
"""

import base64
import sys
from pathlib import Path

import pytest

from tests.stubs.box_sdk import setup_box_modules


def _import_handler():
    """Import and reload BoxHandler after patching modules."""
    import importlib
    MODULE_DIR = Path(__file__).resolve().parents[2] / 'code' / 'modules'
    sys.path.append(str(MODULE_DIR))
    module = importlib.import_module('BoxHandler')
    return importlib.reload(module).BoxHandler


def test_authenticate_service(tmp_path, monkeypatch):
    """
    Purpose:
        Ensure authentication succeeds using credentials from a single
        environment variable.
    Inputs:
        tmp_path (Path): Temporary directory for isolating test artifacts.
        monkeypatch: Pytest fixture to modify environment and modules.
    Outputs:
        None; assertions validate that the client object is created.
    """
    captured = {}
    setup_box_modules(monkeypatch, captured)
    monkeypatch.chdir(tmp_path)

    BoxHandler = _import_handler()

    secret = base64.b64encode(b'{}').decode()
    monkeypatch.setenv('SECRET_ENV', secret)

    handler = BoxHandler({'STORAGE_COMMON': {'secret_key_name': 'SECRET_ENV'}, 'BOX': {}})
    handler._authenticate()

    assert handler.client == 'client'
    assert captured['config'] == {}


def test_missing_primary_secret_raises(tmp_path, monkeypatch):
    """
    Purpose:
        Verify that omitting ``secret_key_name`` from the configuration triggers
        a ``KeyError`` as the primary credential is mandatory.
    Inputs:
        tmp_path (Path): Temporary directory for test isolation.
        monkeypatch: Fixture used to stub box modules.
    Outputs:
        None; the test passes if ``KeyError`` is raised.
    """
    captured = {}
    setup_box_modules(monkeypatch, captured)
    monkeypatch.chdir(tmp_path)

    BoxHandler = _import_handler()

    handler = BoxHandler({'STORAGE_COMMON': {}, 'BOX': {}})
    with pytest.raises(KeyError):
        handler._authenticate()


def test_authenticate_split_env(tmp_path, monkeypatch):
    """Ensure credentials split across two env vars are concatenated."""
    captured = {}
    setup_box_modules(monkeypatch, captured)
    monkeypatch.chdir(tmp_path)

    BoxHandler = _import_handler()

    secret = base64.b64encode(b'{}').decode()
    first, second = secret[: len(secret)//2], secret[len(secret)//2 :]
    monkeypatch.setenv('SECRET_ENV', first)
    monkeypatch.setenv('SECRET_ENV2', second)

    handler = BoxHandler({
        'STORAGE_COMMON': {
            'secret_key_name': 'SECRET_ENV',
            'secret_key_name_2': 'SECRET_ENV2',
        },
        'BOX': {},
    })
    handler._authenticate()

    assert handler.client == 'client'
    assert captured['config'] == {}
