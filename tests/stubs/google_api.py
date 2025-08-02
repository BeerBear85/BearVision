import sys
import types
import re


class DummyCreds:
    pass


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


def install_google_stubs():
    """Insert dummy google modules into sys.modules."""
    sys.modules.setdefault('google', types.ModuleType('google'))

    apiclient_mod = types.ModuleType('googleapiclient')
    disc_mod = types.ModuleType('googleapiclient.discovery')
    http_mod = types.ModuleType('googleapiclient.http')
    disc_mod.build = lambda *a, **k: None
    http_mod.MediaFileUpload = DummyMediaFileUpload
    http_mod.MediaIoBaseDownload = DummyMediaIoBaseDownload

    sys.modules['googleapiclient'] = apiclient_mod
    sys.modules['googleapiclient.discovery'] = disc_mod
    sys.modules['googleapiclient.http'] = http_mod


class FakeDriveService:
    """Minimal in-memory fake of the Google Drive service."""

    def __init__(self):
        self.store = {}
        self.next_id = 1

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


def setup_google_modules(monkeypatch, captured_info):
    """Patch google modules using the monkeypatch fixture."""
    install_google_stubs()
    import googleapiclient.discovery as disc_mod
    import googleapiclient.http as http_mod

    def fake_build(*a, **k):
        captured_info['creds'] = k.get('credentials')
        return 'service'

    monkeypatch.setattr(disc_mod, 'build', fake_build, raising=False)
    monkeypatch.setattr(http_mod, 'MediaFileUpload', DummyMediaFileUpload, raising=False)
    monkeypatch.setattr(http_mod, 'MediaIoBaseDownload', DummyMediaIoBaseDownload, raising=False)

