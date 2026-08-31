# ADR 0006: Windows and Linux Edge computers

Status: accepted

BearVision 3 supports Windows and 64-bit Linux Edge computers. Both platforms
run the same Python runtime, configuration contracts and behavioural scenarios;
the thin Node.js control server and React GUI are optional operator interfaces.
Raspberry Pi OS on 64-bit ARM is the first documented Linux deployment.

Platform-specific provisioning and process supervision are allowed, but must
remain outside the runtime modules. Windows and Linux are verified in CI, while
each physical device profile must still be validated with its BLE adapter,
GoPro networking, YOLO inference and FFmpeg installation before production use.

The specific computer model remains a benchmark decision rather than an
architecture constraint.
