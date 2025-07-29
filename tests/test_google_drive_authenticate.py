import json
import os
import sys
import base64
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / 'code' / 'modules'
sys.path.append(str(MODULE_DIR))

from tests.stubs.google_api import setup_google_modules


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

