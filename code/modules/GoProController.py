import asyncio
import threading
from pathlib import Path
import logging
from open_gopro import WirelessGoPro, models
from open_gopro.models.constants import settings
from open_gopro.models.streaming import StreamType, PreviewStreamOptions
from open_gopro.models.constants import constants

logger = logging.getLogger(__name__)


class GoProController:
    """Simplified wrapper around the OpenGoPro API."""

    def __init__(self, target: str | None = None) -> None:
        self._gopro = WirelessGoPro(target=target)
        self._loop = None
        self._loop_thread = None

    def _run_in_thread(self, coro):
        """Run a coroutine in a separate thread with its own event loop."""
        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        
        result_container = []
        exception_container = []
        
        def thread_target():
            try:
                result = run()
                result_container.append(result)
            except Exception as e:
                exception_container.append(e)
        
        thread = threading.Thread(target=thread_target)
        thread.start()
        thread.join()
        
        if exception_container:
            raise exception_container[0]
        
        return result_container[0] if result_container else None

    def connect(self) -> None:
        """Open connection to the GoPro."""
        self._run_in_thread(self._gopro.open())

    def disconnect(self) -> None:
        """Close connection to the GoPro."""
        self._run_in_thread(self._gopro.close())

    def list_videos(self) -> list[str]:
        """Return list of video filenames stored on the camera."""
        resp = self._run_in_thread(self._gopro.http_command.get_media_list())
        return [f.filename for f in resp.data.files]

    def download_file(self, camera_file: str, local_path: str) -> Path:
        """Download a specific file from the camera."""
        resp = self._run_in_thread(
            self._gopro.http_command.download_file(
                camera_file=camera_file, local_file=Path(local_path)
            )
        )
        return resp.data

    def configure(self) -> None:
        """Configure the GoPro for BearVision usage."""
        self._run_in_thread(self._configure())

    async def _configure(self) -> None:
        await self._gopro.http_command.load_preset_group(
            group=models.proto.EnumPresetGroup.PRESET_GROUP_ID_VIDEO
        )
        await self._gopro.http_settings.video_resolution.set(
            settings.VideoResolution.NUM_4K
        )
        await self._gopro.http_settings.frames_per_second.set(
            settings.FramesPerSecond.NUM_60_0
        )
        await self._gopro.http_settings.hindsight.set(
            settings.Hindsight.NUM_15_SECONDS
        )

    def start_preview(self, port: int = 8554) -> str:
        """Start preview stream and return its URL."""
        options = PreviewStreamOptions(port=port)
        self._run_in_thread(self._gopro.streaming.start_stream(StreamType.PREVIEW, options))
        assert self._gopro.streaming.url is not None
        return self._gopro.streaming.url

    def start_hindsight_clip(self, duration: float = 1.0) -> None:
        """Trigger a HindSight capture on the camera."""
        self._run_in_thread(self._start_hindsight_clip(duration))

    async def _start_hindsight_clip(self, duration: float) -> None:
        await self._gopro.http_command.set_shutter(shutter=constants.Toggle.ENABLE)
        await asyncio.sleep(duration)
        await self._gopro.http_command.set_shutter(shutter=constants.Toggle.DISABLE)
