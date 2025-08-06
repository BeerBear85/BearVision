import os


def test_google_credentials_b64_defined_and_not_unset():
    """
    Purpose:
        Verify that the ``GOOGLE_CREDENTIALS_B64`` environment variable is set
        to a meaningful value.
    Inputs:
        None.
    Outputs:
        None; assertions fail if the variable is missing or carries the sentinel
        value ``UNSET``.
    """
    value = os.getenv('GOOGLE_CREDENTIALS_B64')
    assert value is not None and value != '', 'GOOGLE_CREDENTIALS_B64 should be set'
    assert value != 'UNSET', 'GOOGLE_CREDENTIALS_B64 should not be UNSET'
