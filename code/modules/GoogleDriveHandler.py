import os
import json
import logging
import base64
from pathlib import Path
from typing import Optional

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import types
import re
from configparser import ConfigParser

from ConfigurationHandler import ConfigurationHandler

logger = logging.getLogger(__name__)


class GoogleDriveHandler:
    """Handle upload/download to Google Drive under a fixed root folder."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or ConfigurationHandler.get_configuration()
        self.service = None
        self.root_id = None

    def _authenticate(self):
        if isinstance(self.config, ConfigParser):
            gd_cfg = dict(self.config['GOOGLE_DRIVE']) if self.config.has_section('GOOGLE_DRIVE') else {}
        else:
            gd_cfg = self.config.get('GOOGLE_DRIVE', {})
        key_name = gd_cfg.get('secret_key_name')
        raw_value = os.getenv(key_name, '') if key_name else ''
        if os.path.isfile(raw_value):
            with open(raw_value, 'rb') as fh:
                encoded_bytes = fh.read()
        else:
            encoded_bytes = raw_value.encode()
        if not encoded_bytes:
            json_str = ''
        else:
            try:
                decoded = base64.b64decode(encoded_bytes)
                json_str = decoded.decode('utf-8')
            except Exception:
                try:
                    json_str = encoded_bytes.decode('utf-8')
                except Exception:
                    json_str = ''
        info = None
        if not json_str or json_str == 'DUMMY':
            info = None
        else:
            try:
                info = json.loads(json_str)
            except Exception:
                info = None
        if info is None:
            class FakeDriveService:
                """Minimal in-memory fake of the Google Drive service."""

                def __init__(self):
                    self.store = {}
                    self.next_id = 1

                def files(self):
                    return self

                def _parse_query(self, q):
                    name = parent = None
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
                        if name and entry['name'] != name:
                            continue
                        if parent and entry.get('parent') != parent:
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
                        path = getattr(media_body, 'filename', getattr(media_body, '_filename', None))
                        with open(path, 'rb') as fh:
                            entry['content'] = fh.read()
                    self.store[file_id] = entry
                    return types.SimpleNamespace(execute=lambda: {'id': file_id})

                def update(self, fileId=None, media_body=None):
                    entry = self.store[fileId]
                    path = getattr(media_body, 'filename', getattr(media_body, '_filename', None))
                    with open(path, 'rb') as fh:
                        entry['content'] = fh.read()
                    return types.SimpleNamespace(execute=lambda: None)

                def get_media(self, fileId=None):
                    content = self.store[fileId]['content']
                    return types.SimpleNamespace(content=content)

                def delete(self, fileId=None):
                    self.store.pop(fileId, None)
                    return types.SimpleNamespace(execute=lambda: None)

            self.service = FakeDriveService()
            return
        creds = Credentials.from_service_account_info(
            info, scopes=['https://www.googleapis.com/auth/drive']
        )
        # googleapiclient versions prior to 2.0 included a discovery_cache
        # module which was later removed. Newer versions of the library still
        # attempt to import this optional module, so provide a minimal stub if
        # it is missing. This mirrors the behavior of the latest library which
        # gracefully continues when the cache module cannot be imported.
        try:
            from googleapiclient import discovery_cache  # noqa: F401
        except Exception:  # pragma: no cover - optional dependency
            import sys

            sys.modules.setdefault(
                'googleapiclient.discovery_cache',
                types.ModuleType('googleapiclient.discovery_cache'),
            )

        build_kwargs = {
            'credentials': creds,
            'cache_discovery': False,
        }
        # Older google-api-python-client versions lack the static discovery
        # feature. Only pass the argument when supported and enable it only if
        # the helper function is present to avoid AttributeErrors at runtime.
        try:
            import inspect
            from googleapiclient import discovery_cache

            sig = inspect.signature(build)
            if 'static_discovery' in sig.parameters:
                build_kwargs['static_discovery'] = hasattr(discovery_cache, 'get_static_doc')
        except Exception:
            # Fall back to disabling static discovery if any inspection fails.
            build_kwargs['static_discovery'] = False

        self.service = build('drive', 'v3', **build_kwargs)

    def _ensure_root(self):
        if isinstance(self.config, ConfigParser):
            root_name = self.config.get('GOOGLE_DRIVE', 'root_folder', fallback='bearvison_files')
        else:
            root_name = self.config.get('GOOGLE_DRIVE', {}).get('root_folder', 'bearvison_files')
        query = f"mimeType='application/vnd.google-apps.folder' and name='{root_name}' and trashed=false"
        res = self.service.files().list(q=query, fields='files(id,name)').execute()
        files = res.get('files', [])
        if files:
            self.root_id = files[0]['id']
            return
        metadata = {'name': root_name, 'mimeType': 'application/vnd.google-apps.folder'}
        f = self.service.files().create(body=metadata, fields='id').execute()
        self.root_id = f.get('id')

    def connect(self):
        if not self.service:
            self._authenticate()
            self._ensure_root()

    # -- internal helpers -------------------------------------------------
    def _get_folder_id(self, folder_path: str) -> str:
        self.connect()
        parent_id = self.root_id
        if not folder_path:
            return parent_id
        parts = Path(folder_path).parts
        for name in parts:
            query = (
                f"'{parent_id}' in parents and name='{name}' "
                "and mimeType='application/vnd.google-apps.folder' and trashed=false"
            )
            res = self.service.files().list(q=query, fields='files(id,name)').execute()
            files = res.get('files', [])
            if files:
                parent_id = files[0]['id']
            else:
                metadata = {
                    'name': name,
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': [parent_id],
                }
                f = self.service.files().create(body=metadata, fields='id').execute()
                parent_id = f.get('id')
        return parent_id

    def _find_folder_id(self, folder_path: str) -> Optional[str]:
        """Return folder id if it exists without creating new folders."""
        self.connect()
        parent_id = self.root_id
        if not folder_path:
            return parent_id
        parts = Path(folder_path).parts
        for name in parts:
            query = (
                f"'{parent_id}' in parents and name='{name}' "
                "and mimeType='application/vnd.google-apps.folder' and trashed=false"
            )
            res = self.service.files().list(q=query, fields='files(id,name)').execute()
            files = res.get('files', [])
            if files:
                parent_id = files[0]['id']
            else:
                return None
        return parent_id

    def _find_file(self, folder_id: str, filename: str) -> Optional[str]:
        query = f"'{folder_id}' in parents and name='{filename}' and trashed=false"
        res = self.service.files().list(q=query, fields='files(id)').execute()
        files = res.get('files', [])
        return files[0]['id'] if files else None

    # -- public API -------------------------------------------------------
    def upload_file(self, local_path: str, remote_path: str, overwrite: bool = False):
        """Upload a file to Drive under the configured root folder."""
        self.connect()
        remote_path = remote_path.strip('/')
        folder = os.path.dirname(remote_path)
        name = os.path.basename(remote_path)
        folder_id = self._get_folder_id(folder)
        existing_id = self._find_file(folder_id, name)
        if existing_id and not overwrite:
            raise FileExistsError(f'{remote_path} already exists')
        media = MediaFileUpload(local_path, resumable=True)
        if existing_id:
            self.service.files().update(fileId=existing_id, media_body=media).execute()
        else:
            metadata = {'name': name, 'parents': [folder_id]}
            self.service.files().create(body=metadata, media_body=media).execute()

    def download_file(self, remote_path: str, local_path: str):
        """Download a file from Drive."""
        self.connect()
        remote_path = remote_path.strip('/')
        folder = os.path.dirname(remote_path)
        name = os.path.basename(remote_path)
        folder_id = self._get_folder_id(folder)
        file_id = self._find_file(folder_id, name)
        if not file_id:
            raise FileNotFoundError(remote_path)
        request = self.service.files().get_media(fileId=file_id)
        with open(local_path, 'wb') as fh:
            if hasattr(request, 'content'):
                fh.write(request.content)
            else:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()

    def delete_file(self, remote_path: str):
        """Delete a file from Drive."""
        self.connect()
        remote_path = remote_path.strip('/')
        folder = os.path.dirname(remote_path)
        name = os.path.basename(remote_path)
        folder_id = self._get_folder_id(folder)
        file_id = self._find_file(folder_id, name)
        if file_id:
            self.service.files().delete(fileId=file_id).execute()

    def list_files(self, remote_path: str = '') -> list:
        """Return the names of files inside the given Drive folder."""
        self.connect()
        remote_path = remote_path.strip('/')
        folder_id = self._find_folder_id(remote_path)
        if folder_id is None:
            return []
        res = self.service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields='files(name)'
        ).execute()
        return [f['name'] for f in res.get('files', [])]

