# ADR 0005: Box storage for version 3.0

Status: accepted

Box is both the Edge/server transport and result store. There is no direct HTTP
API between the runtimes. A job uploads under `input-queue/uploading`, writes
READY last, then moves as a folder to `input-queue/ready`. The worker claims by
moving to `processing` and finishes under `processed/user_<uuid>`, `unresolved` or
`failed`. One provider-neutral lifecycle implements these transitions over a
small folder-store seam; Box and the deterministic filesystem supply only the
seven generic list, existence, read, download, write, move and delete
operations. The in-memory simulator follows the same queue contract. Email is
mutable contact data and is never used as a storage path or permanent identity.
