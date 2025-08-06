import os
import base64
import json
from pathlib import Path
from typing import Optional
import logging
from configparser import ConfigParser

from boxsdk import JWTAuth, Client

from ConfigurationHandler import ConfigurationHandler

logger = logging.getLogger(__name__)


class BoxHandler:
    """Handle upload/download to Box under a fixed root folder."""

    def __init__(self, config: Optional[dict] = None):
        """
        Purpose:
            Initialize the handler with Box configuration.
        Inputs:
            config (Optional[dict]): Mapping or ``ConfigParser`` with a
                ``BOX`` section. When omitted, configuration is loaded
                via :class:`ConfigurationHandler`.
        Outputs:
            None
        """
        self.config = config or ConfigurationHandler.get_configuration()
        self.client: Optional[Client] = None
        self.root_id: Optional[str] = None

    def _authenticate(self):
        """
        Purpose:
            Build an authenticated Box client using credentials stored in one or
            two base64-encoded environment variables. Splitting allows very long
            secrets to bypass OS limits.
        Inputs:
            None directly; relies on configuration values ``secret_key_name`` and
            optionally ``secret_key_name_2`` inside the ``STORAGE_COMMON``
            section. Service specific values from ``BOX`` are still honored to
            allow overrides.
        Outputs:
            None; ``self.client`` is set on success.
        """
        if isinstance(self.config, ConfigParser):
            common_cfg = (
                dict(self.config["STORAGE_COMMON"])
                if self.config.has_section("STORAGE_COMMON")
                else {}
            )
            box_cfg = (
                dict(self.config["BOX"])
                if self.config.has_section("BOX")
                else {}
            )
        else:
            common_cfg = self.config.get("STORAGE_COMMON", {})
            box_cfg = self.config.get("BOX", {})

        secret_env = common_cfg.get("secret_key_name") or box_cfg.get("secret_key_name")
        if not secret_env:
            # Centralizing the secret name in STORAGE_COMMON avoids duplication,
            # but we still fail fast when it is missing so deployments discover
            # misconfiguration immediately.
            raise KeyError(
                "Missing 'secret_key_name' in STORAGE_COMMON or BOX configuration"
            )

        secret_env_2 = (
            common_cfg.get("secret_key_name_2")
            or box_cfg.get("secret_key_name_2", "")  # allow service overrides
        )

        secret_b64 = os.getenv(secret_env, "")
        if secret_env_2:
            # Concatenating the two pieces keeps the decoding logic simple while
            # still allowing deployments to split secrets across variables.
            secret_b64 += os.getenv(secret_env_2, "")
        if not secret_b64:
            raise RuntimeError(
                f"Box credentials not found. Set the '{secret_env}' environment variable."
            )

        try:
            info = json.loads(base64.b64decode(secret_b64).decode("utf-8"))
            auth = JWTAuth.from_settings_dictionary(info)
            self.client = Client(auth)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("Invalid Box credentials") from exc

    def _ensure_root(self):
        """
        Purpose:
            Resolve or create the root folder used as the base for all
            operations.
        Inputs:
            None.
        Outputs:
            None; ``self.root_id`` is set to the root folder's ID.
        """
        if isinstance(self.config, ConfigParser):
            root_name = self.config.get("BOX", "root_folder", fallback="bearvison_files")
        else:
            root_name = self.config.get("BOX", {}).get("root_folder", "bearvison_files")

        root = self.client.folder("0")
        for item in root.get_items():
            if item.type == "folder" and item.name == root_name:
                self.root_id = item.id
                return
        new_folder = root.create_subfolder(root_name)
        self.root_id = new_folder.id

    def connect(self):
        """
        Purpose:
            Lazily establish a connection to Box before performing operations.
        Inputs:
            None.
        Outputs:
            None.
        """
        if not self.client:
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
            str: Box folder ID.
        """
        self.connect()
        parent_id = self.root_id
        if not folder_path:
            return parent_id
        for name in Path(folder_path).parts:
            folder = self.client.folder(parent_id)
            for item in folder.get_items():
                if item.type == "folder" and item.name == name:
                    parent_id = item.id
                    break
            else:
                created = folder.create_subfolder(name)
                parent_id = created.id
        return parent_id

    def _find_folder_id(self, folder_path: str) -> Optional[str]:
        """
        Purpose:
            Retrieve the ID for an existing folder path without creating new
            folders.
        Inputs:
            folder_path (str): Slash-separated path relative to the root folder.
        Outputs:
            Optional[str]: Folder ID if found, otherwise ``None``.
        """
        self.connect()
        parent_id = self.root_id
        if not folder_path:
            return parent_id
        for name in Path(folder_path).parts:
            folder = self.client.folder(parent_id)
            for item in folder.get_items():
                if item.type == "folder" and item.name == name:
                    parent_id = item.id
                    break
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
        folder = self.client.folder(folder_id)
        for item in folder.get_items():
            if item.type == "file" and item.name == filename:
                return item.id
        return None

    # -- public API -------------------------------------------------------
    def upload_file(self, local_path: str, remote_path: str, overwrite: bool = False):
        """
        Purpose:
            Upload a local file into Box under the configured root folder.
        Inputs:
            local_path (str): Path to the local file.
            remote_path (str): Destination path inside Box relative to the
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
        if existing_id:
            self.client.file(existing_id).update_contents(local_path)
        else:
            self.client.folder(folder_id).upload(local_path, name)

    def download_file(self, remote_path: str, local_path: str):
        """
        Purpose:
            Download a file from Box to the local filesystem.
        Inputs:
            remote_path (str): Path within Box relative to the root folder.
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
        with open(local_path, "wb") as fh:
            self.client.file(file_id).download_to(fh)

    def delete_file(self, remote_path: str):
        """
        Purpose:
            Remove a file from Box if it exists.
        Inputs:
            remote_path (str): Path to the file inside Box relative to the
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
            self.client.file(file_id).delete()

    def list_files(self, remote_path: str = "") -> list:
        """
        Purpose:
            List the names of files within a Box folder.
        Inputs:
            remote_path (str): Folder path inside Box relative to the root
                folder. Defaults to the root folder when empty.
        Outputs:
            list: Collection of filenames contained in the target folder.
        """
        self.connect()
        remote_path = remote_path.strip("/")
        folder_id = self._find_folder_id(remote_path)
        if folder_id is None:
            return []
        return [item.name for item in self.client.folder(folder_id).get_items() if item.type == "file"]
