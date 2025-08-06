import os
import pytest


@pytest.mark.skipif(
    os.getenv('STORAGE_CREDENTIALS_B64') in (None, '', 'UNSET'),
    reason='STORAGE_CREDENTIALS_B64 not set',
)
def test_storage_credentials_b64_defined_and_not_unset():
    """
    Purpose:
        Verify that the ``STORAGE_CREDENTIALS_B64`` environment variable is set
        to a meaningful value.
    Inputs:
        None.
    Outputs:
        None; assertions fail if the variable is missing or carries the sentinel
        value ``UNSET``.
    """
    value = os.getenv('STORAGE_CREDENTIALS_B64')
    assert value is not None and value != '', 'STORAGE_CREDENTIALS_B64 should be set'
    assert value != 'UNSET', 'STORAGE_CREDENTIALS_B64 should not be UNSET'
