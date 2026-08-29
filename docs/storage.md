# Box Storage Handler

BearVision currently stores generated files in Box. The versioned edge
configuration selects the provider:

```yaml
config_schema_version: "2.0"
config_kind: bearvision-edge
storage:
  provider: box
```

The handler uses the environment variable `STORAGE_CREDENTIALS_B64` (and the
optional `STORAGE_CREDENTIALS_B64_2`) to supply base64 encoded authentication
information:

```yaml
storage:
  root_folder: bearvision_files
  credential_env: STORAGE_CREDENTIALS_B64
  secondary_credential_env: STORAGE_CREDENTIALS_B64_2
```

The `BoxHandler` provides `upload_file`, `download_file`, `delete_file`, and
`list_files` methods. It operates relative to a configurable root folder and
lazily establishes its network connection for simplicity during testing.

## Integration test

The focused storage test exercises both the in-memory substitute and the real
Box adapter without running the complete recording workflow:

```bash
uv sync --locked --extra dev
uv run pytest tests/cloud/test_box_drive.py -v
```

The simulated case always runs. The real case runs when
`STORAGE_CREDENTIALS_B64` (and, when configured, its second part) is available;
otherwise pytest reports that case as skipped. The uploaded test object is
given a unique name and deleted again after the round trip.
