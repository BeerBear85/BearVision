# Box Storage Handler

BearVision currently stores generated files in Box. The active storage service
is selected in `config.ini` under the `[WEB_STORIES]` section:

```ini
[WEB_STORIES]
storage_service = box
```

The handler uses the environment variable `STORAGE_CREDENTIALS_B64` (and the
optional `STORAGE_CREDENTIALS_B64_2`) to supply base64 encoded authentication
information:

```ini
[STORAGE_COMMON]
secret_key_name = STORAGE_CREDENTIALS_B64
secret_key_name_2 = STORAGE_CREDENTIALS_B64_2

[BOX]
root_folder = bearvision_files
```

The `BoxHandler` provides `upload_file`, `download_file`, `delete_file`, and
`list_files` methods. It operates relative to a configurable root folder and
lazily establishes its network connection for simplicity during testing.
