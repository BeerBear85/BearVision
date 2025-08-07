"""Stubs for the :mod:`boxsdk` package used in the test-suite.

The real Box SDK is fairly heavy and would pull in a large dependency tree as
well as perform network operations.  The tests only need to verify that our
code interacts with a *Box-like* API, so we provide a tiny in-memory
implementation plus a few helpers to monkeypatch behaviour.  Keeping everything
in this single module avoids scattering mocks throughout the tests and makes the
intent explicit.
"""

import sys
from unittest.mock import MagicMock


class DummyAuth:
    """Placeholder object representing authentication credentials.

    Purpose:
        Act as a simple stand‑in for the object returned by the real SDK after
        authentication.  No behaviour is required; the object merely needs to be
        identifiable.
    Inputs:
        None.
    Outputs:
        None.  Instances are used purely for identity.
    """

    # No methods are required – the object only serves as a unique token.
    pass


class DummyJWTAuth:
    """Minimal stand‑in for :class:`boxsdk.JWTAuth`.

    Purpose:
        Provide the class structure expected by the code under test so that we
        can monkeypatch its behaviour.  Only the factory method used in the
        tests is implemented.
    Inputs:
        None when instantiated directly; the configuration is supplied via the
        class method below.
    Outputs:
        Instances of :class:`DummyAuth` representing authentication credentials.
    """

    @classmethod
    def from_settings_dictionary(cls, cfg):
        """Create a dummy auth object from a configuration dictionary.

        Purpose:
            Mirror the interface of the real method so that higher‑level code
            can call it without modification.
        Inputs:
            cfg (dict): Dictionary of configuration settings.  The values are
                ignored but captured by tests to validate the call.
        Outputs:
            DummyAuth: Placeholder authentication object.
        """

        # Returning a simple object keeps the stub lightweight while still
        # providing something the calling code can hold on to.
        return DummyAuth()


class DummyClient:
    """Basic client stub used for authentication tests.

    Purpose:
        Mimic the :class:`boxsdk.Client` class sufficiently for tests that only
        check whether authentication occurs.
    Inputs:
        auth (DummyAuth): Authentication token that would be passed to the real
            client.
    Outputs:
        None.  The resulting object simply stores the token for later
        inspection.
    """

    def __init__(self, auth):
        """Store the provided authentication token.

        Purpose:
            Retain the token so tests can assert it was passed correctly.
        Inputs:
            auth (DummyAuth): Placeholder authentication object.
        Outputs:
            None.
        """

        # Keeping only the auth attribute avoids implementing the entire client
        # API yet allows tests to ensure the token is wired through.
        self.auth = auth


class FakeItem:
    """Represent a file or folder in the fake Box store.

    Purpose:
        Provide a minimal data container used by the in-memory API below.
    Inputs:
        item_id (str): Unique identifier for the object.
        name (str): Display name of the item.
        item_type (str): Either ``"file"`` or ``"folder"`` to mirror Box's
            terminology.
        parent (str | None): Identifier of the parent folder. ``None`` denotes
            the root.
        content (bytes | None): File contents if the item represents a file.
    Outputs:
        None.  Instances simply expose the attributes for tests to inspect.
    """

    def __init__(self, item_id, name, item_type, parent=None, content=None):
        self.id = item_id
        self.name = name
        self.type = item_type
        self.parent = parent
        self.content = content


class FakeBoxClient:
    """In-memory fake implementing a tiny subset of the Box API.

    Purpose:
        Allow tests to exercise higher-level logic that expects to interact with
        a Box client without performing I/O or network calls.
    Inputs:
        None when instantiated.
    Outputs:
        FakeBoxClient: Object exposing ``folder`` and ``file`` accessors.
    """

    def __init__(self):
        """Initialize the client with a root folder.

        Purpose:
            Using a dictionary-based store keeps operations O(1) and trivial to
            reason about, which is ideal for tests.
        Inputs:
            None.
        Outputs:
            None.
        """

        self.store = {"0": FakeItem("0", "root", "folder")}
        self.next_id = 1

    # --- folder handling -------------------------------------------------
    def folder(self, folder_id):
        """Return a wrapper for folder operations.

        Purpose:
            Provide a fluent interface similar to the real SDK where actions are
            performed on objects retrieved from the client.
        Inputs:
            folder_id (str): Identifier of the folder to wrap.
        Outputs:
            FakeFolder: Object exposing folder-like behaviour.
        """

        return FakeFolder(self, folder_id)

    # --- file handling ---------------------------------------------------
    def file(self, file_id):
        """Return a wrapper for file operations.

        Purpose:
            Mirror the pattern used by the actual SDK for symmetry with folder
            handling.
        Inputs:
            file_id (str): Identifier of the file to wrap.
        Outputs:
            FakeFile: Object exposing file-like behaviour.
        """

        return FakeFile(self, file_id)


class FakeFolder:
    """Wrapper providing folder-like operations on the in-memory store.

    Purpose:
        Give tests a small yet expressive API to interact with folders.
    Inputs:
        client (FakeBoxClient): The owning client instance.
        folder_id (str): Identifier of the folder represented by this wrapper.
    Outputs:
        FakeFolder: The constructed wrapper.
    """

    def __init__(self, client, folder_id):
        self.client = client
        self.folder_id = folder_id

    def get_items(self):
        """List items directly contained in the folder.

        Purpose:
            Mimic ``folder.get_items`` from the real SDK to allow simple
            traversal in tests.
        Inputs:
            None.
        Outputs:
            list[FakeItem]: Items whose ``parent`` matches this folder.
        """

        return [item for item in self.client.store.values() if item.parent == self.folder_id]

    def create_subfolder(self, name):
        """Create a new subfolder and return it.

        Purpose:
            Provide minimal mutation behaviour so tests can verify folder
            creation logic.
        Inputs:
            name (str): Desired folder name.
        Outputs:
            FakeItem: Representation of the created folder.
        """

        item_id = f"id{self.client.next_id}"
        self.client.next_id += 1
        item = FakeItem(item_id, name, "folder", parent=self.folder_id)
        self.client.store[item_id] = item
        return item

    def upload(self, local_path, name):
        """Upload a file into this folder.

        Purpose:
            Simulate file uploads without hitting a remote service.
        Inputs:
            local_path (str): Path to the source file on disk.
            name (str): Target name within the folder.
        Outputs:
            FakeItem: Representation of the uploaded file.
        """

        item_id = f"id{self.client.next_id}"
        self.client.next_id += 1
        with open(local_path, "rb") as fh:
            content = fh.read()
        item = FakeItem(item_id, name, "file", parent=self.folder_id, content=content)
        self.client.store[item_id] = item
        return item


class FakeFile:
    """Wrapper providing file-like operations on the in-memory store.

    Purpose:
        Expose enough of the file API for higher-level code to update and
        retrieve contents.
    Inputs:
        client (FakeBoxClient): The owning client instance.
        file_id (str): Identifier of the file represented by this wrapper.
    Outputs:
        FakeFile: The constructed wrapper.
    """

    def __init__(self, client, file_id):
        self.client = client
        self.file_id = file_id

    def update_contents(self, local_path):
        """Replace the file's contents with data from ``local_path``.

        Purpose:
            Allow tests to emulate uploading a new version of an existing file.
        Inputs:
            local_path (str): Path to the local file whose contents should be
                stored.
        Outputs:
            None.
        """

        with open(local_path, "rb") as fh:
            self.client.store[self.file_id].content = fh.read()

    def download_to(self, fh):
        """Write the file's contents to the provided file handle.

        Purpose:
            Simulate downloading without performing network I/O.
        Inputs:
            fh (BinaryIO): Open file-like object to receive the data.
        Outputs:
            None.
        """

        fh.write(self.client.store[self.file_id].content)

    def delete(self):
        """Remove the file from the in-memory store.

        Purpose:
            Allow tests to verify deletion logic without touching a real API.
        Inputs:
            None.
        Outputs:
            None.
        """

        self.client.store.pop(self.file_id, None)


# -- helper functions for tests ------------------------------------------

def install_box_stubs():
    """Install lightweight Box SDK stubs into :mod:`sys.modules`.

    Purpose:
        Ensure imports of ``boxsdk`` within the code under test resolve to our
        controlled stand-ins instead of attempting to load the real dependency.
    Inputs:
        None.
    Outputs:
        None.
    """

    # ``MagicMock`` is used instead of a real module object because it accepts
    # any attribute access, which keeps the stub flexible and avoids having to
    # implement every part of the SDK we don't care about.
    boxsdk_mock = MagicMock()
    boxsdk_mock.JWTAuth = DummyJWTAuth
    boxsdk_mock.Client = DummyClient

    # Overwrite any real Box SDK installation to keep tests hermetic.  By
    # inserting into ``sys.modules`` we influence subsequent ``import boxsdk``
    # statements throughout the test run.
    sys.modules["boxsdk"] = boxsdk_mock


def setup_box_modules(monkeypatch, captured):
    """Patch Box SDK modules using the :mod:`pytest` ``monkeypatch`` fixture.

    Purpose:
        Replace the network‑dependent parts of the Box SDK with controllable
        fakes while capturing the values passed to them.  This allows tests to
        assert on configuration and authentication behaviour without contacting
        external services.
    Inputs:
        monkeypatch: Fixture used to dynamically patch objects during tests.
        captured (dict): Dictionary that will receive the captured ``config`` and
            ``auth`` values.
    Outputs:
        None.
    """

    install_box_stubs()
    import boxsdk

    def fake_from_settings_dictionary(cls, cfg):
        """Record configuration and return a dummy auth object.

        Purpose:
            Allow tests to inspect the configuration dictionary passed to the
            SDK's factory method.
        Inputs:
            cls (type): Ignored; present for classmethod signature compliance.
            cfg (dict): Configuration to be recorded.
        Outputs:
            DummyAuth: Placeholder auth object used by the client.
        """

        captured["config"] = cfg
        return DummyAuth()

    def fake_client(auth):
        """Record the provided auth object and return a simple identifier.

        Purpose:
            Let tests verify that authentication objects are passed through
            correctly when instantiating the client.
        Inputs:
            auth (DummyAuth): Authentication token produced earlier.
        Outputs:
            str: Constant value representing the fake client.
        """

        captured["auth"] = auth
        return "client"

    # We patch methods rather than replacing whole objects to minimise the
    # surface area of our test doubles while still allowing the production code
    # to interact with objects that look realistic.
    monkeypatch.setattr(
        boxsdk.JWTAuth,
        "from_settings_dictionary",
        classmethod(fake_from_settings_dictionary),
        raising=False,
    )
    monkeypatch.setattr(boxsdk, "Client", fake_client, raising=False)
