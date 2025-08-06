import sys
import types


class DummyAuth:
    """Placeholder object representing authentication credentials."""
    pass


class DummyJWTAuth:
    """Minimal JWTAuth stub capturing config for inspection."""

    @classmethod
    def from_settings_dictionary(cls, cfg):  # pragma: no cover - simple stub
        return DummyAuth()


class DummyClient:
    """Basic client stub used for authentication tests."""

    def __init__(self, auth):  # pragma: no cover - simple stub
        self.auth = auth


class FakeItem:
    """Represent a file or folder in the fake Box store."""

    def __init__(self, item_id, name, item_type, parent=None, content=None):
        self.id = item_id
        self.name = name
        self.type = item_type
        self.parent = parent
        self.content = content


class FakeBoxClient:
    """In-memory fake implementing a tiny subset of the Box API."""

    def __init__(self):
        self.store = {"0": FakeItem("0", "root", "folder")}
        self.next_id = 1

    # --- folder handling -------------------------------------------------
    def folder(self, folder_id):
        return FakeFolder(self, folder_id)

    # --- file handling ---------------------------------------------------
    def file(self, file_id):
        return FakeFile(self, file_id)


class FakeFolder:
    def __init__(self, client, folder_id):
        self.client = client
        self.folder_id = folder_id

    def get_items(self):
        return [item for item in self.client.store.values() if item.parent == self.folder_id]

    def create_subfolder(self, name):
        item_id = f"id{self.client.next_id}"
        self.client.next_id += 1
        item = FakeItem(item_id, name, "folder", parent=self.folder_id)
        self.client.store[item_id] = item
        return item

    def upload(self, local_path, name):
        item_id = f"id{self.client.next_id}"
        self.client.next_id += 1
        with open(local_path, "rb") as fh:
            content = fh.read()
        item = FakeItem(item_id, name, "file", parent=self.folder_id, content=content)
        self.client.store[item_id] = item
        return item


class FakeFile:
    def __init__(self, client, file_id):
        self.client = client
        self.file_id = file_id

    def update_contents(self, local_path):
        with open(local_path, "rb") as fh:
            self.client.store[self.file_id].content = fh.read()

    def download_to(self, fh):
        fh.write(self.client.store[self.file_id].content)

    def delete(self):
        self.client.store.pop(self.file_id, None)


# -- helper functions for tests ------------------------------------------

def install_box_stubs():
    """Insert dummy boxsdk modules into sys.modules."""
    mod = types.ModuleType("boxsdk")
    mod.JWTAuth = DummyJWTAuth
    mod.Client = DummyClient
    # Overwrite any real boxsdk installation to keep tests hermetic
    sys.modules["boxsdk"] = mod


def setup_box_modules(monkeypatch, captured):
    """Patch boxsdk modules using the monkeypatch fixture."""
    install_box_stubs()
    import boxsdk

    def fake_from_settings_dictionary(cls, cfg):
        captured["config"] = cfg
        return DummyAuth()

    def fake_client(auth):
        captured["auth"] = auth
        return "client"

    monkeypatch.setattr(boxsdk.JWTAuth, "from_settings_dictionary", classmethod(fake_from_settings_dictionary), raising=False)
    monkeypatch.setattr(boxsdk, "Client", fake_client, raising=False)
