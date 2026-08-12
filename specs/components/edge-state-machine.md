# Edge state machine specification

## States

`initializing → looking → capturing → processing → uploading → looking`

Any active state may enter `recovering` after a retryable component failure or
`stopped` after a terminal failure or shutdown request.

## Rules

- Capture requires a person detection but never derives identity from it.
- A BLE assignment may be assigned, unassigned or ambiguous.
- Repeated commands use stable request identifiers and must be idempotent.
- All timeouts use monotonic time supplied by the configured clock.
- Uploaded media remains provider-neutral outside the storage adapter.
