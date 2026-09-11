"""Regression tests for native read/release ownership during cancellation."""
import asyncio
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from bearvision.adapters import OpenCvPreviewFrameSource, SystemClock
from bearvision.ports import ComponentUnavailable


def install_capture(monkeypatch, capture):
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        VideoCapture=capture, CAP_FFMPEG=1900,
        CAP_PROP_OPEN_TIMEOUT_MSEC=53, CAP_PROP_READ_TIMEOUT_MSEC=54,
    ))


@pytest.mark.parametrize("cancel_close", [False, True])
def test_close_never_releases_during_native_read(monkeypatch, cancel_close):
    reading, finish_read, released = Event(), Event(), Event()
    overlaps = []

    class Capture:
        def __init__(self, *args):
            pass

        def isOpened(self):
            return True

        def read(self):
            reading.set()
            finish_read.wait(timeout=2)
            reading.clear()
            return False, None

        def release(self):
            overlaps.append(reading.is_set())
            released.set()

    install_capture(monkeypatch, Capture)

    async def exercise():
        source = OpenCvPreviewFrameSource(SystemClock())
        await source.open("udp://preview")
        close = None
        try:
            assert await asyncio.to_thread(reading.wait, 1)
            close = asyncio.create_task(source.close())
            await asyncio.sleep(0.05)
            assert not released.is_set(), "release raced with native read"
            if cancel_close:
                close.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await close
            finish_read.set()
            if not cancel_close:
                await asyncio.wait_for(close, 1)
            await asyncio.wait_for(source.close(), 1)
            assert released.is_set()
            assert overlaps == [False]
        finally:
            finish_read.set()
            if close is not None:
                await asyncio.gather(close, return_exceptions=True)
            await source.close()

    asyncio.run(exercise())


def test_cancelled_open_releases_late_native_capture(monkeypatch):
    opening, finish_open, released = Event(), Event(), Event()

    class Capture:
        def __init__(self, *args):
            opening.set()
            finish_open.wait(timeout=2)

        def isOpened(self):
            return True

        def release(self):
            released.set()

    install_capture(monkeypatch, Capture)

    async def exercise():
        source = OpenCvPreviewFrameSource(SystemClock())
        task = asyncio.create_task(source.open("udp://preview"))
        try:
            assert await asyncio.to_thread(opening.wait, 1)
            task.cancel()
            finish_open.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(source.close(), 1)
            assert released.is_set(), "cancelled open leaked its late capture"
        finally:
            finish_open.set()
            await asyncio.gather(task, return_exceptions=True)
            await source.close()

    asyncio.run(exercise())


def test_native_read_failure_reaches_frame_consumer_and_releases(monkeypatch):
    released = Event()
    constructor_args = []

    class Capture:
        def __init__(self, *args):
            constructor_args.append(args)

        def isOpened(self):
            return True

        def read(self):
            return False, None

        def release(self):
            released.set()

    install_capture(monkeypatch, Capture)

    async def exercise():
        source = OpenCvPreviewFrameSource(SystemClock())
        await source.open("udp://preview")
        try:
            with pytest.raises(ComponentUnavailable, match="stopped producing frames"):
                await asyncio.wait_for(anext(source.frames()), 0.5)
        finally:
            await source.close()
        assert released.is_set()

    asyncio.run(exercise())
    _, backend, params = constructor_args[0]
    assert backend == 1900
    assert 0 < dict(zip(params[::2], params[1::2]))[53] <= 20000
    assert 0 < dict(zip(params[::2], params[1::2]))[54] <= 1000
