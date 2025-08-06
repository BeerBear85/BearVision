# Storage Handlers

BearVision can store generated files in either Google Drive or Box. The active
storage service is selected in `config.ini` under the `[WEB_STORIES]` section:

```ini
[WEB_STORIES]
storage_service = google_drive  # or "box"
```

Both services use the environment variable `STORAGE_CREDENTIALS_B64` (and the
optional `STORAGE_CREDENTIALS_B64_2`) to supply base64 encoded authentication
information. The variable names are defined once in a shared configuration
section to avoid duplication:

```ini
[STORAGE_COMMON]
secret_key_name = STORAGE_CREDENTIALS_B64
secret_key_name_2 = STORAGE_CREDENTIALS_B64_2

[GOOGLE_DRIVE]
root_folder = bearvisson_files

[BOX]
root_folder = bearvision_files
```

The `BoxHandler` mirrors the existing `GoogleDriveHandler` and provides
`upload_file`, `download_file`, `delete_file`, and `list_files` methods. Both
handlers operate relative to a configurable root folder and lazily establish
network connections for simplicity during testing.
