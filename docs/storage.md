# Box Storage Handler

BearVision currently stores generated files in Box. The versioned edge
configuration selects the provider:

```yaml
config_schema_version: "1.0"
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
