import base64
import pickle
import sys
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / 'code' / 'modules'
sys.path.append(str(MODULE_DIR))

from tests.stubs.google_api import setup_google_modules, DummyCreds


def test_authenticate_uses_token_and_env(tmp_path, monkeypatch):
    captured = {}
    setup_google_modules(monkeypatch, captured)
    monkeypatch.chdir(tmp_path)

    import importlib
    module = importlib.import_module('GoogleDriveHandler')
    module = importlib.reload(module)
    GoogleDriveHandler = module.GoogleDriveHandler

    token_creds = DummyCreds()
    with open('token.pickle', 'wb') as fh:
        pickle.dump(token_creds, fh)

    secret = base64.b64encode(b'{}').decode()
    monkeypatch.setenv('GOOGLE_OAUTH_CLIENT_SECRET_B64', secret)

    handler = GoogleDriveHandler({'GOOGLE_DRIVE': {}})
    handler._authenticate()

    assert handler.service == 'service'
    assert isinstance(captured['creds'], DummyCreds)
    assert Path('credentials.json').exists()

