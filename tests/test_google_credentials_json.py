import os


def test_google_credentials_json_defined_and_not_unset():
    value = os.getenv('GOOGLE_CREDENTIALS_JSON')
    assert value is not None and value != '', 'GOOGLE_CREDENTIALS_JSON should be set'
    assert value != 'UNSET', 'GOOGLE_CREDENTIALS_JSON should not be UNSET'
