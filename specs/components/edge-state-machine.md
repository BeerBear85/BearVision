# Edge state machine specification

## States

`initializing → monitoring → recording → packaging → uploading → monitoring`

Any active state may enter `recovering` after a retryable component failure or
`stopped` after a terminal failure or shutdown request.

## Rules

- Capture requires a person detection but never derives identity from it.
- Capture starts immediately after person detection and runs for the configured
  fixed duration.
- `Camera.capture` returns a playable asset containing precisely the requested
  capture window. A recorded-video camera performs the cut locally on the Edge
  computer before job packaging and upload.
- Edge includes every BearTag observation from the complete clip as a relative
  millisecond offset plus RSSI and acceleration in m/s².
- Edge never receives the user registry, calculates scores or chooses a rider.
- Repeated commands use stable request identifiers and must be idempotent.
- All timeouts use monotonic time supplied by the configured clock.
- READY is committed only after manifest, video and NDJSON have been uploaded.
- Queue operations remain provider-neutral outside the queue adapter.
- Source media is immutable. Clip output is committed atomically; failed
  partial output must be removed.
