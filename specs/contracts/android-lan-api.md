# Android LAN API

Status: prototype.

The Android API is a separate read-only listener on `0.0.0.0:4321`. The
administrative interface remains bound to `127.0.0.1:4320`.

Every user-specific request carries the claimed normalized email in the
`X-BearVision-Email` header. There is no proof of ownership in this prototype,
so the API is suitable only for non-sensitive test data on a trusted LAN.

## Endpoints

- `GET /api/app/health`
- `GET /api/app/videos?page=1&pageSize=50`
- `GET /api/app/videos/<job-id>/thumbnail`
- `GET /api/app/videos/<job-id>/video`

The list response contains only the registered display name and public video
metadata. Media is returned only if the processed job belongs to the email in
the request header. Video responses support HTTP byte ranges for seeking.

The LAN listener can be overridden with `BEARVISION_APP_HOST` and
`BEARVISION_APP_PORT`.
