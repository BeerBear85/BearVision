import os
import json
import logging
from pathlib import Path
from typing import Optional

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from ConfigurationHandler import ConfigurationHandler

logger = logging.getLogger(__name__)


class GoogleDriveHandler:
    """Handle upload/download to Google Drive under a fixed root folder."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or ConfigurationHandler.get_configuration()
        self.service = None
        self.root_id = None

    def _authenticate(self):
        gd_cfg = self.config.get('GOOGLE_DRIVE', {})
        key_name = gd_cfg.get('secret_key_name')
        json_str = os.getenv(key_name, '') if key_name else ''
        if not json_str:
            raise RuntimeError(f'Missing credential json in environment variable {key_name}')
        info = json.loads(json_str)
        creds = Credentials.from_service_account_info(info, scopes=['https://www.googleapis.com/auth/drive'])
        self.service = build('drive', 'v3', credentials=creds)

    def _ensure_root(self):
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

