# Virtual cameraman

## Responsibility

Reduce an extracted clip before cloud upload while keeping the rider inside a
stable crop. The processor runs on Edge before the anonymous job package is
committed; it has no assignment or user identity input.

## Observable layers

- Green bounding box: a YOLO person measurement. It does not establish rider
  identity.
- Red cross: the Kalman + RTS smoothed rider-position estimate.
- Red circle: a conservative circular 95% region derived from the largest
  eigenvalue of the two-dimensional position covariance.
- Cyan rectangle: the separate zero-phase Butterworth camera path and the crop
  written to the processed upload file.

## Current implementation

- Detector sampling: 10 Hz over the whole extracted clip.
- State: image `x`, `y`, `vx`, `vy` with damped constant velocity. The process
  noise permits rapid image-plane acceleration, while the default five-second
  velocity damping time constant preserves fast cross-frame motion and still
  brings stale tracks gradually to rest.
- Estimation: Kalman forward pass with normalized-innovation gating, followed
  by an RTS backward pass that uses future detections to refine past states.
- Track acquisition: the first two plausible measurements initialize image
  velocity before normal innovation gating. This prevents a fast rider from
  being rejected solely because the initial velocity starts at zero. After a
  detection gap, a measurement within the configured maximum image-plane speed
  can reacquire the track when the covariance gate alone rejects it.
- Length adjustment: after estimating the rider path for the complete source
  clip, retain at most one second before its first in-frame position and one
  second after its last in-frame position. This happens before crop rendering.
- Camera motion: second-order Butterworth filtering forward and backward, so
  smoothing introduces no phase delay.
- Output: silent 160 x 90 H.264 for the 320 x 180 recorded-video scenario.
- Engineering artefacts: annotated video and frame-level JSON.
- Upload ordering: raw extraction, tracking/crop, validation, job packaging,
  then READY commit to the cloud queue.

## Limits

The 95% label describes the filter's Gaussian model, not demonstrated empirical
coverage. Process and measurement noise must be calibrated against labelled
field trajectories before the number can be treated as a field-validated 95%
guarantee. Multi-person association is also not solved by this component.
