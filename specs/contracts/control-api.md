# Edge Control API 1.0

Status: initial implementation.

The Node.js Edge Control server is a process supervisor and presentation
adapter. It must not implement BearVision domain policy.

## HTTP

- `GET /api/health` returns the selected mode and runtime phase.
- `GET /api/scenarios` lists supported versioned scenario files.
- `POST /api/mode` selects `simulation` or `hardware` while idle.
- `POST /api/run` starts the selected runtime.
- `POST /api/stop` requests termination of the active runtime.
- `GET /api/events` streams version 1.0 control events as Server-Sent Events.

Python emits a discriminated, versioned runtime-event envelope before Node
assigns transport sequence and wall-clock emission metadata. Every runtime
event has `control_event_version: "1.0"`, a known `kind`, a typed payload and a
non-negative scenario `at_s` or `null`. Node rejects malformed envelopes as
runtime log output instead of forwarding partially understood domain events.

The API is intended for the co-located React GUI on the Edge computer. Remote
network access, authentication and authorization are not specified yet.
