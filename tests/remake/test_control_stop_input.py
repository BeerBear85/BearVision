"""A stop command must finish even when the supervisor keeps stdin open."""
import json
import subprocess
import sys


def test_stop_command_does_not_wait_for_another_input_line():
    program = """
import asyncio
from bearvision.control import _read_control_commands
async def main():
    stop = asyncio.Event()
    await _read_control_commands(object(), shutdown_requested=stop)
    assert stop.is_set()
print('ready', flush=True)
asyncio.run(main())
"""
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", program],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        child.stdin.write(json.dumps({"command_version": "1.0", "kind": "stop_runtime"}) + "\n")
        child.stdin.flush()
        # Do not communicate() or close stdin: that would hide the production bug.
        assert child.wait(timeout=3) == 0
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)
