import os
import logging
import base64
import json
from pathlib import Path
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.errors import HttpError
from google.oauth2 import service_account
from configparser import ConfigParser

from ConfigurationHandler import ConfigurationHandler

logger = logging.getLogger(__name__)


class GoogleDriveHandler:
    """Handle upload/download to Google Drive under a fixed root folder."""

    def __init__(self, config: Optional[dict] = None):
        """
        Purpose:
            Initialize the handler with Drive configuration.
        Inputs:
            config (Optional[dict]): Mapping or ``ConfigParser`` with a
                ``GOOGLE_DRIVE`` section. When omitted, configuration is loaded
                via :class:`ConfigurationHandler`.
        Outputs:
            None
        """
        self.config = config or ConfigurationHandler.get_configuration()
        self.service = None
        self.root_id = None

    def _authenticate(self):
        """
        Purpose:
            Build an authenticated Google Drive service using credentials
            stored in one or two base64-encoded environment variables. Splitting
            allows extremely long credential strings to be handled without
            exceeding environment limits. Depending on the configuration, either
            an OAuth 2.0 user flow or a service account is used.
        Inputs:
            None; relies on environment variables and instance configuration.
            ``secret_key_name`` is required in the ``GOOGLE_DRIVE`` section and
            names the primary environment variable. ``secret_key_name_2`` is
            optional and, when provided, holds the name of a secondary variable
            containing the remaining credential characters.
        Outputs:
            None; sets ``self.service`` when successful.
        """
        if isinstance(self.config, ConfigParser):
            gd_cfg = (
                dict(self.config["GOOGLE_DRIVE"])
                if self.config.has_section("GOOGLE_DRIVE")
                else {}
            )
        else:
            gd_cfg = self.config.get("GOOGLE_DRIVE", {})

        try:
            secret_env = gd_cfg["secret_key_name"]
            # Fail fast if configuration omits the primary secret name so
            # missing credentials are caught during startup rather than at
            # runtime when authentication is attempted.
        except KeyError as exc:
            raise KeyError(
                "Missing 'secret_key_name' in GOOGLE_DRIVE configuration"
            ) from exc

        try:
            secret_env_2 = gd_cfg["secret_key_name_2"]
            # The second environment variable is optional; when present it lets
            # deployments split credentials to bypass shell length limits.
        except KeyError:
            # If not provided, treat as empty string so later concatenation does
            # not need special cases.
            secret_env_2 = ""

        auth_mode = gd_cfg.get("auth_mode", "user").lower()

        # Concatenate the two environment variables. Splitting allows very large
        # base64 strings to bypass shell-specific length limits while keeping the
        # decoding logic simple.
        secret_b64 = os.getenv(secret_env, "")
        if secret_env_2:
            # Only attempt to load the secondary part when configured.
            secret_b64 += os.getenv(secret_env_2, "")
        if not secret_b64:
            raise RuntimeError(
                f"Google Drive credentials not found. Set the '{secret_env}' environment variable.",
            )

        try:
            # Credentials are decoded in memory to avoid writing temporary
            # files which may leak secrets or require cleanup.
            info = json.loads(base64.b64decode(secret_b64).decode("utf-8"))

            if auth_mode == "service":
                # Service accounts are ideal for headless or automated
                # environments where a user is not present.
                creds = service_account.Credentials.from_service_account_info(
                    info,
                    scopes=["https://www.googleapis.com/auth/drive"],
                )
            else:
                # OAuth flow is chosen to allow end-users to grant access from
                # their own Google accounts. A dynamic port avoids collisions
                # on multi-user systems.
                flow = InstalledAppFlow.from_client_config(
                    info, scopes=["https://www.googleapis.com/auth/drive.file"]
                )
                creds = flow.run_local_server(port=0)
                # Persist the token for convenience so subsequent runs can
                # reuse the refresh token instead of prompting every time.
                with open("token.json", "w") as token:
                    token.write(creds.to_json())
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("Invalid Google Drive credentials") from exc

        build_kwargs = {
            "credentials": creds,
            "cache_discovery": False,  # disable to avoid local cache dependency
        }
        try:
            self.service = build("drive", "v3", **build_kwargs)
        except HttpError as error:
            # Network or API errors are surfaced to aid diagnostics rather than
            # silently ignoring failures.
            print(f"An error occurred: {error}")



    def _ensure_root(self):
        """
        Purpose:
            Resolve or create the root folder used as the base for all operations.
        Inputs:
            None.
        Outputs:
            None; ``self.root_id`` is set to the root folder's ID.
        """
        if isinstance(self.config, ConfigParser):
            root_name = self.config.get(
                "GOOGLE_DRIVE", "root_folder", fallback="bearvison_files"
            )
        else:
            root_name = self.config.get("GOOGLE_DRIVE", {}).get(
                "root_folder", "bearvison_files"
            )
        query = (
            f"mimeType='application/vnd.google-apps.folder' and name='{root_name}' and trashed=false"
        )
        res = self.service.files().list(q=query, fields="files(id,name)").execute()
        files = res.get("files", [])
        if files:
            self.root_id = files[0]["id"]
            return
        metadata = {"name": root_name, "mimeType": "application/vnd.google-apps.folder"}
        f = self.service.files().create(body=metadata, fields="id").execute()
        self.root_id = f.get("id")

    def connect(self):
        """
        Purpose:
            Ensure a connection to Google Drive is established before any operation.
            The call is lazy so instantiating the handler has no side effects.
        Inputs:
            None.
        Outputs:
            None.
        """
        if not self.service:
            self._authenticate()
            self._ensure_root()

    # -- internal helpers -------------------------------------------------
    def _get_folder_id(self, folder_path: str) -> str:
        """
        Purpose:
            Resolve the ID of the given folder path, creating intermediate
            folders as needed under the configured root.
        Inputs:
            folder_path (str): Slash-separated path relative to the root folder.
        Outputs:
            str: Google Drive folder ID.
        """
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
            res = self.service.files().list(q=query, fields="files(id,name)").execute()
            files = res.get("files", [])
            if files:
                parent_id = files[0]["id"]
            else:
                metadata = {
                    "name": name,
                    "mimeType": "application/vnd.google-apps.folder",
                    "parents": [parent_id],
                }
                f = self.service.files().create(body=metadata, fields="id").execute()
                parent_id = f.get("id")
        return parent_id

    def _find_folder_id(self, folder_path: str) -> Optional[str]:
        """
        Purpose:
            Retrieve the ID for an existing folder path without creating new
            folders.
        Inputs:
            folder_path (str): Slash-separated path relative to the root
                folder.
        Outputs:
            Optional[str]: Folder ID if the path exists, otherwise ``None``.
        """
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
            res = self.service.files().list(q=query, fields="files(id,name)").execute()
            files = res.get("files", [])
            if files:
                parent_id = files[0]["id"]
            else:
                return None
        return parent_id

    def _find_file(self, folder_id: str, filename: str) -> Optional[str]:
        """
        Purpose:
            Locate a file by name within a folder.
        Inputs:
            folder_id (str): Parent folder ID.
            filename (str): Name of the file to search for.
        Outputs:
            Optional[str]: File ID if found, otherwise ``None``.
        """
        query = f"'{folder_id}' in parents and name='{filename}' and trashed=false"
        res = self.service.files().list(q=query, fields="files(id)").execute()
        files = res.get("files", [])
        return files[0]["id"] if files else None

    # -- public API -------------------------------------------------------
    def upload_file(self, local_path: str, remote_path: str, overwrite: bool = False):
        """
        Purpose:
            Upload a local file into Drive under the configured root folder.
        Inputs:
            local_path (str): Path to the local file.
            remote_path (str): Destination path inside Drive relative to the
                root folder.
            overwrite (bool): Replace an existing file if set to ``True``.
        Outputs:
            None.
        """
        self.connect()
        remote_path = remote_path.strip("/")
        folder = os.path.dirname(remote_path)
        name = os.path.basename(remote_path)
        folder_id = self._get_folder_id(folder)
        existing_id = self._find_file(folder_id, name)
        if existing_id and not overwrite:
            raise FileExistsError(f"{remote_path} already exists")
        media = MediaFileUpload(local_path, resumable=True)
        if existing_id:
            self.service.files().update(fileId=existing_id, media_body=media).execute()
        else:
            metadata = {"name": name, "parents": [folder_id]}
            self.service.files().create(body=metadata, media_body=media).execute()

    def download_file(self, remote_path: str, local_path: str):
        """
        Purpose:
            Download a file from Drive to the local filesystem.
        Inputs:
            remote_path (str): Path within Drive relative to the root folder.
            local_path (str): Destination path on the local filesystem.
        Outputs:
            None.
        """
        self.connect()
        remote_path = remote_path.strip("/")
        folder = os.path.dirname(remote_path)
        name = os.path.basename(remote_path)
        folder_id = self._get_folder_id(folder)
        file_id = self._find_file(folder_id, name)
        if not file_id:
            raise FileNotFoundError(remote_path)
        request = self.service.files().get_media(fileId=file_id)
        with open(local_path, "wb") as fh:
            if hasattr(request, "content"):
                fh.write(request.content)
            else:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()

    def delete_file(self, remote_path: str):
        """
        Purpose:
            Remove a file from Drive if it exists.
        Inputs:
            remote_path (str): Path to the file inside Drive relative to the
                root folder.
        Outputs:
            None.
        """
        self.connect()
        remote_path = remote_path.strip("/")
        folder = os.path.dirname(remote_path)
        name = os.path.basename(remote_path)
        folder_id = self._get_folder_id(folder)
        file_id = self._find_file(folder_id, name)
        if file_id:
            self.service.files().delete(fileId=file_id).execute()

    def list_files(self, remote_path: str = "") -> list:
        """
        Purpose:
            List the names of files within a Drive folder.
        Inputs:
            remote_path (str): Folder path inside Drive relative to the root
                folder. Defaults to the root folder when empty.
        Outputs:
            list: Collection of filenames contained in the target folder.
        """
        self.connect()
        remote_path = remote_path.strip("/")
        folder_id = self._find_folder_id(remote_path)
        if folder_id is None:
            return []
        res = self.service.files().list(
            q=f"'{folder_id}' in parents and trashed=false", fields="files(name)"
        ).execute()
        return [f["name"] for f in res.get("files", [])]
