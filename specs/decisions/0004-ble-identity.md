# ADR 0004: Whole-clip BearTag rider identity

Status: accepted

BearVision 3 assigns rider identity only from registered BearTag observations.
Each BearTag supplies RSSI and three-axis acceleration at approximately 10 Hz.
Vision triggers a fixed-duration clip when it detects a person; it is not
required to determine the jump timestamp or rider identity.

For the complete interval from detection until the configured clip end, the
assignment policy:

1. Computes orientation-independent activity for every observation as the
   acceleration magnitude's deviation from standard gravity.
2. Averages all activity measurements from the clip for each tag. Signed X/Y/Z
   axes are not averaged because motion in opposite directions would cancel.
3. Uses median RSSI over the same clip to reduce single-packet radio noise.
4. Requires a configurable minimum sample count plus motion and RSSI gates.
5. Ranks qualifying tags using a configurable weighted score, initially 70%
   motion and 30% RSSI.
6. Returns `ambiguous` when the leading scores are inside the configured margin.

Capture begins immediately on detection. Assignment and upload happen after the
clip closes, when all BearTag observations from that clip are available. The
initial thresholds and weights remain configurable assumptions.
