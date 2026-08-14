# Server job contract version 1

```text
input-queue/ready/<job-id>/
  manifest.json
  video.mp4
  beartag-data.ndjson
  READY
```

`manifest.json` contains `schemaVersion`, `jobId`, `edgeDeviceId`, `createdAt`,
`captureStartedAt`, `captureEndedAt`, and a video object with `filename`,
`mimeType`, `sizeBytes` and lowercase SHA-256. All timestamps are timezone-aware
UTC instants.

Each NDJSON line contains `bearTagId`, integer `offsetMs` from clip start,
`rssiDbm`, and `accelerationMps2` with `x`, `y`, `z`. User identity is forbidden
from the package. The server validates paths, interval bounds, byte length and
checksum before scoring.

Terminal jobs contain `result.json` with status, processing time, algorithm
version, selected tag/user/assignment when applicable, every candidate score,
reason and error code. READY-less folders are ignored and `jobId` is the
idempotency key.
