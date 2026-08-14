"""Stable user identity helpers shared by queue adapters."""

from __future__ import annotations

from uuid import UUID


USER_FOLDER_PREFIX = "user_"


def user_storage_folder(user_id: UUID | str) -> str:
    """Return the only allowed storage folder representation for a user."""

    return f"{USER_FOLDER_PREFIX}{UUID(str(user_id))}"


def user_id_from_storage_folder(folder: str) -> UUID:
    """Decode and validate a canonical user storage folder."""

    if not folder.startswith(USER_FOLDER_PREFIX):
        raise ValueError("invalid user storage folder")
    user_id = UUID(folder.removeprefix(USER_FOLDER_PREFIX))
    if folder != user_storage_folder(user_id):
        raise ValueError("user storage folder is not canonical")
    return user_id
