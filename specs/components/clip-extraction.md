# Edge clip-extraction specification

## Responsibility

The Windows Edge computer converts a source video and `CaptureRequest` into a
new local MP4. The camera adapter owns this operation; the orchestrator remains
independent of FFmpeg and file formats.

## Behaviour

- Start is `requested_at_monotonic_s - pre_roll_s`, clamped to zero.
- Duration is `pre_roll_s + post_roll_s`.
- Version 3.0 currently issues `pre_roll_s: 0`.
- The source file must remain byte-for-byte unchanged.
- Output is H.264 video with AAC audio when the source contains audio.
- Output is written as `.partial.mp4` and atomically renamed only after probe
  validation succeeds.
- A repeated request ID is idempotent and returns the existing validated clip.
- File names derive only from the request ID and never from unchecked paths.
- Failure removes partial output and raises a typed component error.

## Timing tolerance

Regression output must start within one source frame of the requested start and
have the requested duration within 100 milliseconds, except when the source
ends before the requested window.
