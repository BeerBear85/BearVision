import base64
import pickle
import sys
from pathlib import Path
import tempfile

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
    monkeypatch.setenv('SECRET_ENV', secret)

    paths = []
    orig_ntf = tempfile.NamedTemporaryFile

    def fake_ntf(*args, **kwargs):
        kwargs.setdefault('delete', False)
        tmp = orig_ntf(*args, **kwargs)
        paths.append(Path(tmp.name))
        return tmp

    monkeypatch.setattr(module.tempfile, 'NamedTemporaryFile', fake_ntf)

    handler = GoogleDriveHandler({'GOOGLE_DRIVE': {'secret_key_name': 'SECRET_ENV'}})
    handler._authenticate()

    assert handler.service == 'service'
    assert isinstance(captured['creds'], DummyCreds)
    assert paths and not paths[0].exists()

