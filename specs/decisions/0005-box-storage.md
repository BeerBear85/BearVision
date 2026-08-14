# ADR 0005: Box storage for version 3.0

Status: accepted

Box is both the Edge/server transport and result store. There is no direct HTTP
API between the runtimes. A job uploads under `input-queue/uploading`, writes
READY last, then moves as a folder to `input-queue/ready`. The worker claims by
moving to `processing` and finishes under `processed/<email>`, `unresolved` or
`failed`. Provider-neutral queue ports are implemented by Box and deterministic
filesystem/in-memory adapters.
