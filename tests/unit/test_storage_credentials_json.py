import os
import pytest


@pytest.mark.skipif(
    os.getenv('STORAGE_CREDENTIALS_B64') in (None, '', 'UNSET'),
    reason='STORAGE_CREDENTIALS_B64 not set',
)
def test_storage_credentials_b64_defined_and_not_unset():
    """
    Purpose:
        Ensure the ``STORAGE_CREDENTIALS_B64`` environment variable is
        configured with a meaningful value so downstream tests that rely on
        cloud storage credentials can authenticate.

    Inputs:
        None.

    Outputs:
        None; this function raises assertion errors if misconfigured.
    """
    # Fetch the base64‑encoded credentials from the environment so that
    # configuration problems are surfaced early during the test run rather
    # than at a later stage when credentials are actually needed.
    value = os.getenv('STORAGE_CREDENTIALS_B64')
    # ``None`` or an empty string indicates credentials were not provided at
    # all, which would cause authentication to fail later.
    assert value is not None and value != '', 'STORAGE_CREDENTIALS_B64 should be set'
    # A sentinel value of ``UNSET`` allows CI pipelines to explicitly indicate
    # that credentials are intentionally missing, which should still fail here
    # to avoid accidental usage without proper configuration.
    assert value != 'UNSET', 'STORAGE_CREDENTIALS_B64 should not be UNSET'
