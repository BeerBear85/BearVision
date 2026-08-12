# ADR 0006: Windows Edge computer

Status: accepted

BearVision 3 is deployed on a small Windows computer. The Windows computer
hosts the Python runtime, the thin Node.js control server and the React GUI.

This decision fixes the operating-system integration surface for BLE drivers,
GoPro networking, process supervision, packaging and automatic startup. It does
not change the ports, orchestrator or behavioural simulation contracts.

The specific computer model remains a benchmark decision. It must be validated
with GoPro preview, YOLO inference, BLE scanning and control GUI active at the
same time.
