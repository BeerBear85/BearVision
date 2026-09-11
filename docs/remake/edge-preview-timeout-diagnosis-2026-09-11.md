# Edge hardware start timeout: diagnosis and correction

## Confirmed cause

The preview adapter cancelled its asyncio reader task and then called
`VideoCapture.release()` through another default-executor thread. Cancelling
the task did not interrupt the native `VideoCapture.read()` already in progress.
The two operations therefore ran concurrently on the same capture object.

This overlap was observed on the physical GoPro HERO12 attached to
`bearvisionedge1.local`. Both native calls then remained blocked. The camera
handshake's three-second cleanup timeout fired, but `asyncio.run()` waited for
the outstanding executor threads before returning. Node's independent
75-second preflight watchdog eventually returned `READINESS_TIMEOUT` / HTTP 504.

The exact internal FFmpeg/OpenCV lock was not inspected. The adapter-level
ownership error, blocked native calls, and Python executor shutdown wait were
observed directly.

## Diagnostic evidence collected on 2026-09-10

- One direct `POST /api/runs` without a preceding manual readiness check failed
  with HTTP 504 after 76.28 seconds. The browser and consecutive readiness calls
  are not required to trigger the bug.
- An initial instrumented full preflight completed camera cleanup and BLE.
  Storage checks under the SSH user failed due to that user's permissions;
  those failures are not evidence of a service-user storage problem.
- Of three subsequent camera-only probes, two hung in preview cleanup and one
  completed in 18.79 seconds.
- A native-call trace recorded first-frame completion at 17.655 s, another
  `read()` at 17.656 s, and overlapping `release()` at 17.658 s. Cleanup timed
  out at 20.659 s. GoPro stop and disconnect completed at 21.577 s, but the
  Python executor remained stuck. The bounded probe was terminated externally.
- Changing only cleanup sequencing in a separate diagnostic process produced
  three successful physical probes: 18.821, 18.638 and 18.653 seconds, with no
  read/release overlap. This was an experiment, not the production fix.
- A deterministic local probe against the real adapter also demonstrated
  `release_overlapped_native_read=True`.

## Production correction

`OpenCvPreviewFrameSource` now owns a single-worker executor for the capture.
Opening, inspecting, reading and releasing use that worker. Cleanup is queued
behind native I/O, and cancellation of an async caller does not cancel the
queued release. A capture returned after cancellation during opening is still
released. FFmpeg receives native open/read timeouts of 20 seconds / 1 second
when the capture is created. Read failures reach the frame consumer instead of
leaving it blocked on an empty queue.

## Separate stop issue uncovered during verification

The first deployed preview correction allowed the start preflight to complete
in 24.69 seconds. The initial test driver checked a provisional monitoring
state too early and requested preview before initialization completed; it then
requested stop as cleanup. The runtime emitted `stopped`, but its process did
not exit.

The control-command loop accepted `stop_runtime` and immediately started a new
`asyncio.to_thread(sys.stdin.readline)`. Cancellation could not interrupt that
native read while Node kept stdin open. A real-subprocess regression test
reproduced the hang with stdin deliberately left open. The command loop now
returns after accepting shutdown. The verification driver now waits for an
actual Python `lifecycle_changed: monitoring` event.

## Validation and remaining scope

- All five new regression cases failed before their respective fixes.
- The complete remake suite passed: 222 tests.
- Ruff and mypy passed for the changed production modules.
- Hardware acceptance passed three consecutive manual-readiness, start-preflight,
  confirmed-monitoring, JPEG-preview and graceful-stop cycles through the
  deployed HTTP interface on 2026-09-11:

  | Cycle | Readiness | Start preflight | Runtime initialization | Graceful stop |
  | --- | --- | --- | --- | --- |
  | 1 | 23.12 s, 8/8 | 23.57 s | 20.29 s | 2.19 s |
  | 2 | 23.14 s, 8/8 | 23.64 s | 20.15 s | 2.24 s |
  | 3 | 23.21 s, 8/8 | 23.57 s | 20.04 s | 2.42 s |

  Every cycle had HTTP 200 JPEG preview, zero runtime failures, and returned to
  idle with no active run. No force-stop was used in these acceptance cycles.
- A physical stream-loss probe confirmed the native FFmpeg interrupt at about
  1006 ms and cleanup below one second. An initial six-second probe at one
  consumed frame per second timed out waiting for the consumer error, despite
  successful cleanup. A follow-up at 30 frames per second observed 132 frames
  after preview stop, then the consumer received the native read failure at
  5.737 seconds; cleanup took 0.920 seconds and the process exited normally.
  The one-second setting bounds a native read; buffered frames mean it is not
  a one-second end-to-end stream-loss detection guarantee.
- No camera recording or cloud-upload workflow is claimed verified by those
  lifecycle checks.
- The old readiness result surviving a timeout is a separate server/UI state
  issue. `ReadinessService.run()` retains its previous report when the command
  throws, while the UI timeout notice exists only in component state. This
  correction does not change that behavior.
