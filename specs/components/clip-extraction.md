# Edge clip-extraction specification

## Responsibility

The Camera port converts a `CaptureRequest` into an unmodified raw camera clip
plus timing evidence. The raw clip is the input to Virtual Cameraman and other
processing; it is not the lower-quality preview stream used for detection.

GoPro HindSight is the rolling-buffer mechanism. Pre-roll is the part of a raw
clip before the authoritative detection timestamp. The terms are related but
not interchangeable.

## Behaviour

- The preview frame timestamp is the authoritative detection time.
- Requested start is `detection time - configured HindSight duration`, clamped
  to the earliest media available since HindSight was enabled.
- Requested end is `detection time + post-detection duration`.
- A later detection during an active capture does not extend its end.
- Later person episodes are retained as FIFO requests with their original
  detection timestamps, even while an earlier clip is downloading or processing.
- The downloaded GoPro file remains byte-for-byte unchanged as the raw clip.
- Every capture reports both the requested window and the actual delivered
  window on the monotonic timeline.
- Requested timing is exact. Simulator-delivered timing is exact. Physical
  GoPro timing is explicitly marked estimated because it is derived from camera
  command timing and probed media duration.
- A request whose pre-roll disagrees with the configured GoPro HindSight mode
  fails during capture rather than silently producing a different window.
- After download, enqueue writes only atomic JSON metadata. The raw video remains
  directly under `capture_dir` and is not copied into the queue directory.
- BearTag observations are snapshotted from the actual raw capture window before
  the camera worker accepts another request.

## Physical limitation

Processing and upload are concurrent with preview analysis, but GoPro shutter,
encoding and download remain serial. If their turnaround exceeds the configured
HindSight backlog, the software cannot preserve a later episode; continuous
recording or another capture transport would be a separate design change.

## Recorded-video extraction

- The source file remains byte-for-byte unchanged.
- Output is H.264 video with AAC audio when the source contains audio.
- Output is written as `.partial.mp4` and atomically renamed only after probe
  validation succeeds.
- A repeated request ID is idempotent and returns the existing validated clip.
- File names derive only from the request ID and never from unchecked paths.
- Failure removes partial output and raises a typed component error.

## Timing tolerance

Recorded-video regression output must start within one source frame of the
requested start and have the requested duration within 100 milliseconds, except
when the source begins after the requested start or ends before the requested
end. Physical GoPro duration is accepted only when it lies within the measured
camera-command bounds plus the configured duration tolerance.
