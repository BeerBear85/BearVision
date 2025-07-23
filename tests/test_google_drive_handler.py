import sys
import types
import re
import uuid
from pathlib import Path
from unittest import mock
import pytest

# Create dummy modules so GoogleDriveHandler can be imported without the real
# google packages installed.
sys.modules.setdefault('google', types.ModuleType('google'))

oauth2_mod = types.ModuleType('google.oauth2')
service_account_mod = types.ModuleType('google.oauth2.service_account')
class DummyCreds:
    pass
service_account_mod.Credentials = types.SimpleNamespace(
    from_service_account_info=lambda info, scopes=None: DummyCreds()
)
oauth2_mod.service_account = service_account_mod
sys.modules['google.oauth2'] = oauth2_mod
sys.modules['google.oauth2.service_account'] = service_account_mod

apiclient_mod = types.ModuleType('googleapiclient')
disc_mod = types.ModuleType('googleapiclient.discovery')
http_mod = types.ModuleType('googleapiclient.http')

# Simple placeholders; the real build function isn't used because we patch
# GoogleDriveHandler._authenticate.
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

sys.modules['googleapiclient'] = apiclient_mod
sys.modules['googleapiclient.discovery'] = disc_mod
sys.modules['googleapiclient.http'] = http_mod

# Add module path for imports
MODULE_DIR = Path(__file__).resolve().parents[1] / 'code' / 'modules'
sys.path.append(str(MODULE_DIR))
from GoogleDriveHandler import GoogleDriveHandler


class FakeDriveService:
    """Minimal in-memory fake of the Google Drive service."""

    def __init__(self):
        self.store = {}
        self.next_id = 1

    # The Google API uses a nested files() object. Here we just return self.
    def files(self):
        return self

    def _parse_query(self, q):
        name = None
        parent = None
        is_folder = False
        if q:
            m = re.search(r"name='([^']+)'", q)
            if m:
                name = m.group(1)
            m = re.search(r"'([^']+)' in parents", q)
            if m:
                parent = m.group(1)
            if "mimeType='application/vnd.google-apps.folder'" in q:
                is_folder = True
        return name, parent, is_folder

    def list(self, q=None, fields=None):
        name, parent, is_folder = self._parse_query(q)
        files = []
        for entry in self.store.values():
            if name is not None and entry['name'] != name:
                continue
            if parent is not None and entry.get('parent') != parent:
                continue
            if is_folder and entry['mimeType'] != 'application/vnd.google-apps.folder':
                continue
            files.append({'id': entry['id'], 'name': entry['name']})
        return types.SimpleNamespace(execute=lambda: {'files': files})

    def create(self, body=None, media_body=None, fields=None):
        file_id = f'id{self.next_id}'
        self.next_id += 1
        entry = {
            'id': file_id,
            'name': body.get('name'),
            'mimeType': body.get('mimeType', 'file'),
            'parent': (body.get('parents') or [None])[0],
        }
        if media_body is not None:
            with open(media_body.filename, 'rb') as fh:
                entry['content'] = fh.read()
        self.store[file_id] = entry
        return types.SimpleNamespace(execute=lambda: {'id': file_id})

    def update(self, fileId=None, media_body=None):
        entry = self.store[fileId]
        with open(media_body.filename, 'rb') as fh:
            entry['content'] = fh.read()
        return types.SimpleNamespace(execute=lambda: None)

    def get_media(self, fileId=None):
        content = self.store[fileId]['content']
        return types.SimpleNamespace(content=content)

    def delete(self, fileId=None):
        self.store.pop(fileId, None)
        return types.SimpleNamespace(execute=lambda: None)


@mock.patch.object(GoogleDriveHandler, '_authenticate', lambda self: setattr(self, 'service', FakeDriveService()))
def test_upload_download_delete(tmp_path):
    handler = GoogleDriveHandler({'GOOGLE_DRIVE': {'root_folder': 'root'}})
    handler.connect()

    src = tmp_path / 'src.txt'
    src.write_text('hello')

    handler.upload_file(str(src), 'folder/file.txt')

    dst = tmp_path / 'dst.txt'
    handler.download_file('folder/file.txt', str(dst))
    assert dst.read_text() == 'hello'

    handler.delete_file('folder/file.txt')
    with pytest.raises(FileNotFoundError):
        handler.download_file('folder/file.txt', str(tmp_path / 'missing.txt'))

