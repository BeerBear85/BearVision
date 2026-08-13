# Virtual cameraman

## Responsibility

Reduce an extracted clip before cloud upload while keeping the rider inside a
stable crop. The processor runs on Edge after rider assignment and before the
storage port is called.

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
- State: image `x`, `y`, `vx`, `vy` with damped constant velocity.
- Estimation: Kalman forward pass with normalized-innovation gating, followed
  by an RTS backward pass that uses future detections to refine past states.
- Camera motion: second-order Butterworth filtering forward and backward, so
  smoothing introduces no phase delay.
- Output: 160 x 90 H.264/AAC for the 320 x 180 recorded-video scenario.
- Engineering artefacts: annotated video and frame-level JSON.
- Upload ordering: raw extraction, tracking/crop, validation, then storage.

## Limits

The 95% label describes the filter's Gaussian model, not demonstrated empirical
coverage. Process and measurement noise must be calibrated against labelled
field trajectories before the number can be treated as a field-validated 95%
guarantee. Multi-person association is also not solved by this component.
