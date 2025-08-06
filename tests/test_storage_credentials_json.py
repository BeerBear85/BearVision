import os
import pytest


@pytest.mark.skipif(
    os.getenv('STORAGE_CREDENTIALS_JSON') in (None, '', 'UNSET'),
    reason='STORAGE_CREDENTIALS_JSON not set',
)
def test_storage_credentials_json_defined_and_not_unset():
    value = os.getenv('STORAGE_CREDENTIALS_JSON')
    assert value is not None and value != '', 'STORAGE_CREDENTIALS_JSON should be set'
    assert value != 'UNSET', 'STORAGE_CREDENTIALS_JSON should not be UNSET'
