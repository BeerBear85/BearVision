# Edge state machine specification

## Concurrent state

The live runtime stays `initializing → monitoring → stopping → stopped`.
Camera activity is independent: `idle → capturing → idle`, with an in-memory
FIFO of pending capture requests. The durable background job state is
`queued → processing → packaging → uploading → completed`, with any worker step
able to enter `failed` without changing the live runtime from `monitoring`.

Camera, live detector and writable enqueue storage failures are runtime-critical.
Processing, packaging and upload failures are clip-job failures.

## Rules

- Capture requires a person detection but never derives identity from it.
- A person episode schedules one fixed capture request and returns immediately.
  At least `detection.cooldown_s` without a person separates episodes.
- `Camera.capture` returns a playable asset containing precisely the requested
  capture window. A recorded-video camera performs the cut locally on the Edge
  computer before job packaging and upload.
- Edge includes every BearTag observation from the complete clip as a relative
  millisecond offset plus RSSI and acceleration in m/s².
- Edge never receives the user registry, calculates scores or chooses a rider.
- Repeated commands use stable request identifiers and must be idempotent.
- Only the camera worker calls `Camera.capture`; only one background worker runs
  Virtual Cameraman and publication.
- Queue metadata is atomically persisted under `.raw-clip-queue`; startup moves
  `processing` jobs back to `queued` for at-least-once processing.
- Retry starts processing/packaging failures from raw media and upload failures
  from the retained package checkpoint. There are no automatic job retries.
- All timeouts use monotonic time supplied by the configured clock.
- READY is committed only after manifest, video and NDJSON have been uploaded.
- Queue operations remain provider-neutral outside the queue adapter.
- Source media is immutable. Clip output is committed atomically; failed
  partial output must be removed.
