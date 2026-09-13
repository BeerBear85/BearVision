# BearVision Server Control

Local administration UI for the authoritative Python server worker.

```powershell
cd apps/server-control
corepack pnpm install
corepack pnpm build
$env:BEARVISION_ADMIN_USERNAME = "admin"
$adminPassword = [Convert]::ToBase64String(
  [Security.Cryptography.RandomNumberGenerator]::GetBytes(32)
)
$env:BEARVISION_ADMIN_PASSWORD = $adminPassword
$adminPassword | Set-Clipboard
Remove-Variable adminPassword
corepack pnpm serve
```

Open `http://127.0.0.1:4320`, enter the configured username, and paste the
generated password from the clipboard. Both admin environment variables are
required; Server Control fails closed before its worker or listeners start if
either value is missing or invalid.

Use a unique, randomly generated password. Set credentials only in the process
environment that starts Server Control. Do not store the password in YAML, in a
repository `.env` file, or as a command-line argument. To rotate the password,
set `BEARVISION_ADMIN_PASSWORD` to a newly generated value and restart Server
Control. The admin listener remains restricted to `127.0.0.1:4320`; do not
expose it remotely without a separately designed secure transport and access
layer.

The same process also starts the read-only Android prototype API on port
`4321`, listening on the local network. The Android device must use the
server computer's IPv4 address, for example `http://192.168.1.50:4321`.

## Responsibilities

Python owns:

- paginated job and user read models;
- registry normalization and overlap validation;
- Box and filesystem job discovery;
- media download and SHA-256 verification;
- cached FFmpeg thumbnail generation;
- BearTag scoring and assignment decisions.

Node owns:

- the loopback-only HTTP server;
- Basic authentication in front of the complete admin request handler;
- mapping fixed routes to fixed Python CLI commands;
- static React assets;
- streaming Python-materialized media with HTTP Range support.

Admin credentials are consumed only by Node and are removed from the
environment passed to Python worker and command processes.

The UI provides an overview, a searchable video browser, job details with score
evidence, UUID-preserving user edits, manual clip reassignment, editable BearTag
history with impact preview, and batch recalculation. Manual assignments are
protected unless the operator explicitly includes them. Non-deleting actions do
not request extra confirmation.

Cached media is stored below the configured server `scratch_dir` in
`admin-media/<job-id>/`. The source file is accepted only after its size and
SHA-256 match the versioned job manifest.
