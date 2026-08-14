# ADR 0007: Box-only server processing boundary

Status: accepted

Edge and server communicate only through the versioned Box job package. No
direct HTTP API is introduced. Python remains the single authoritative
implementation of BearTag scoring. The local Node.js server-control component
is an administration/process shell and invokes Python for every registry or
queue mutation, administrative read model and media preparation operation.
Python downloads and verifies video files and generates cached thumbnails with
FFmpeg. Node only serves the React application, maps HTTP requests to explicit
Python CLI commands and streams materialized files with byte-range support.

The first worker processes one job at a time. A claimed `processing` folder is
resumed after restart. Temporary provider failures leave it in place for retry;
permanent package and processing errors receive an explanatory failed result.
Manual requeue is allowed only from `failed` and `unresolved`.

The Edge Control simulation replaces Box with a shared local filesystem queue.
It does not invoke the server worker in-process; a separately started server
runtime consumes the same package contract from `config/server.local.yaml`.
