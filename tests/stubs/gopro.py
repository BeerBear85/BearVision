class DummyResp:
    def __init__(self, data=None, ok=True, status=200):
        self.data = data
        self.ok = ok
        self.status = status


class FakeHttpCommand:
    def __init__(self):
        self.group = None
        self.downloaded = None

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
    def __init__(self):
        self.url = None

    async def start_stream(self, stream_type, options):
        self.url = f"udp://127.0.0.1:{options.port}"
        return DummyResp()


class FakeGoPro:
    def __init__(self, *a, **k):
        self.http_command = FakeHttpCommand()
        self.http_settings = FakeHttpSettings()
        self.streaming = FakeStreaming()

    async def open(self, *a, **k):
        return None

    async def close(self, *a, **k):
        return None

