class DummyResp:
    def __init__(self, data=None, ok=True, status=200):
        self.data = data
        self.ok = ok
        self.status = status


class FakeHttpCommand:
    def __init__(self):
        self.group = None
        self.downloaded = None
        self.shutter = []

    async def get_media_list(self):
        import types
        item = types.SimpleNamespace(filename='DCIM/100GOPRO/GOPR0001.MP4')
        data = types.SimpleNamespace(files=[item])
        return DummyResp(data)

    async def download_file(self, *, camera_file, local_file=None):
        from pathlib import Path
        Path(local_file).write_text('data')
        self.downloaded = (camera_file, str(local_file))
        return DummyResp(Path(local_file))

    async def load_preset_group(self, *, group):
        self.group = group
        return DummyResp()

    async def set_preview_stream(self, *, mode, port=None):
        self.preview = (mode, port)
        # Return response with data containing an ID for wired GoPro compatibility
        data = {'id': f'http://172.24.106.51:8080/gopro/camera/stream/start?port={port}'}
        return DummyResp(data)

    async def set_shutter(self, *, shutter):
        self.shutter.append(shutter)
        return DummyResp()


class FakeSetting:
    def __init__(self):
        self.value = None

    async def set(self, value):
        self.value = value
        return DummyResp()


class FakeHttpSettings:
    def __init__(self):
        self.video_resolution = FakeSetting()
        self.frames_per_second = FakeSetting()
        self.hindsight = FakeSetting()


class FakeStreaming:
    """Serve a video file over HTTP to emulate a live stream."""

    def __init__(self, video_path: str | None = None):
        from pathlib import Path

        self.url = None
        self._server = None
        self._thread = None
        self.video_path = Path(video_path or "tests/data/TestMovie1.mp4")

    def _serve(self, port: int) -> None:
        import http.server
        import socketserver
        import time

        video = self.video_path

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # type: ignore
                if self.path != "/stream":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.end_headers()
                with open(video, "rb") as f:
                    while True:
                        chunk = f.read(8192)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        time.sleep(0.01)

        class ReusableTCPServer(socketserver.TCPServer):
            allow_reuse_address = True

        self._server = ReusableTCPServer(("127.0.0.1", port), Handler)
        self._server.serve_forever()

    async def start_stream(self, stream_type, options):
        import threading

        port = options.port
        self.url = f"http://127.0.0.1:{port}/stream"
        self._thread = threading.Thread(target=self._serve, args=(port,), daemon=True)
        self._thread.start()
        return DummyResp()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1)


class FakeGoPro:
    def __init__(self, *a, **k):
        self.http_command = FakeHttpCommand()
        self.http_settings = FakeHttpSettings()
        self.streaming = FakeStreaming()
        # Set the _serial attribute to simulate a properly opened WiredGoPro
        self._serial = "fake_serial_123"

    async def open(self, *a, **k):
        # Ensure _serial is set when opening
        self._serial = "fake_serial_123"
        return None

    async def close(self, *a, **k):
        return None
    
    @property
    def _base_url(self) -> str:
        """Mock the _base_url property that depends on _serial."""
        if not self._serial:
            from open_gopro.domain.exceptions import GoProNotOpened
            raise GoProNotOpened("Serial / IP has not yet been discovered")
        return f"http://172.24.106.51:8080/gopro"

