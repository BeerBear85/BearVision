# ADR 0004: Server-side whole-clip BearTag identity

Status: accepted

Edge captures every RSSI and acceleration observation in the clip interval and
publishes them anonymously. It does not contain a tag-to-user registry, score a
BearTag, select a winner or construct a user path.

The Python server implementation is authoritative. It retains the established
algorithm: orientation-independent mean `abs(norm(a) - g)`, median RSSI,
sample/motion/RSSI gates, 70/30 default weighting, score margin and deterministic
tag-id ordering. Unassigned and ambiguous selections become `unresolved`.

After exactly one tag wins, the server requires one historical assignment to
cover the complete half-open UTC clip interval. Crossing an assignment boundary
is unresolved. User identity and processed folder names are normalized e-mail
addresses; Gmail dots and `+alias` are never rewritten.
