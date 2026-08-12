# ADR 0001: Ports and adapters

Status: accepted

The BearVision core depends on typed interfaces for time, camera, tag scanning,
detection and storage. Real SDKs and behavioural simulators implement those
interfaces. This allows the same core logic to run in CI and on the edge unit.
