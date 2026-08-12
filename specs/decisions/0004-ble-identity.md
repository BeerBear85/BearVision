# ADR 0004: BLE-only rider identity

Status: accepted

BearVision 3 assigns rider identity only from registered BearTag observations.
Each BearTag supplies both RSSI and three-axis acceleration. Vision may provide
the jump timestamp and trigger recording, but cannot rank or identify riders.

For the configured window around the jump, the assignment policy:

1. Computes orientation-independent motion as the acceleration magnitude's
   deviation from standard gravity.
2. Uses median RSSI to reduce single-packet radio noise.
3. Requires candidates to pass both motion and RSSI gates.
4. Ranks qualifying tags using a configurable weighted score, initially 70%
   motion and 30% RSSI.
5. Returns `ambiguous` when the leading scores are inside the configured margin.

Assignment is deliberately delayed until the configured post-jump window has
elapsed, so accelerometer packets immediately after the visual jump timestamp
are included. GoPro HindSight preserves the preceding media during this delay.

The initial thresholds and weights are hypotheses. They must be calibrated
against labelled BearTag recordings before production use.
