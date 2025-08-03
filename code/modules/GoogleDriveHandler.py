import os
import logging
import base64
import pickle
from pathlib import Path
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import types
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

        secret_env = gd_cfg.get('client_secret_b64_env', 'GOOGLE_OAUTH_CLIENT_SECRET_B64')
        secret_b64 = os.getenv(secret_env, '')

        cred_path = Path('credentials.json')
        if secret_b64 and not cred_path.exists():
            try:
                cred_path.write_bytes(base64.b64decode(secret_b64))
            except Exception:
                cred_path.write_text('')

        creds = None
        token_path = Path('token.pickle')
        if token_path.exists():
            try:
                with token_path.open('rb') as fh:
                    creds = pickle.load(fh)
            except Exception:
                creds = None

        if creds is None:
            raise RuntimeError(
                "Google Drive credentials not found. Set the required environment variables or token file."
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

