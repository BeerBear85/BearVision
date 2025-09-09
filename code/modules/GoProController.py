import asyncio
import threading
from pathlib import Path
import logging
from open_gopro import WiredGoPro, models
from open_gopro.models.constants import settings
from open_gopro.models.constants import constants

logger = logging.getLogger(__name__)


class GoProController:
    """Simplified wrapper around the OpenGoPro API."""

    def __init__(self, target: str | None = None) -> None:
        self._gopro = WiredGoPro(target=target)
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
        # For wired connection, use HTTP command instead of streaming feature
        ip_address = self._run_in_thread(self._start_preview_http(port))
        # Return the UDP stream URL (GoPro wired uses UDP, not RTMP)
        return f"udp://{ip_address}:{port}"
    
    async def _start_preview_http(self, port: int) -> str:
        """Start preview stream using HTTP commands (for wired connection)."""
        # First, stop any existing preview stream to avoid 409 conflict
        try:
            await self._gopro.http_command.set_preview_stream(mode=constants.Toggle.DISABLE)
            await asyncio.sleep(0.5)  # Brief pause to ensure stream stops
        except:
            pass  # Ignore errors if no stream was running
        
        # Start new preview stream
        result = await self._gopro.http_command.set_preview_stream(
            mode=constants.Toggle.ENABLE, port=port
        )
        # Extract IP from the HTTP response URL
        response_url = result.data.get('id', result.id if hasattr(result, 'id') else str(result))
        if '://' in response_url:
            # Extract IP from URL like "http://172.24.106.51:8080/gopro/camera/stream/start?port=8554"
            ip_part = response_url.split('://')[1].split(':')[0]
            return ip_part
        else:
            # Fallback to common GoPro wired IP
            return "172.24.106.51"

    def stop_preview(self) -> None:
        """Stop preview stream."""
        self._run_in_thread(self._stop_preview_http())
    
    async def _stop_preview_http(self) -> None:
        """Stop preview stream using HTTP commands (for wired connection)."""
        await self._gopro.http_command.set_preview_stream(
            mode=constants.Toggle.DISABLE
        )

    def start_hindsight_clip(self, duration: float = 1.0) -> None:
        """Trigger a HindSight capture on the camera."""
        self._run_in_thread(self._start_hindsight_clip(duration))

    async def _start_hindsight_clip(self, duration: float) -> None:
        await self._gopro.http_command.set_shutter(shutter=constants.Toggle.ENABLE)
        await asyncio.sleep(duration)
        await self._gopro.http_command.set_shutter(shutter=constants.Toggle.DISABLE)

    def start_recording(self) -> None:
        """Start video recording on the camera."""
        self._run_in_thread(self._gopro.http_command.set_shutter(shutter=constants.Toggle.ENABLE))

    def stop_recording(self) -> None:
        """Stop video recording on the camera."""
        self._run_in_thread(self._gopro.http_command.set_shutter(shutter=constants.Toggle.DISABLE))

    def get_camera_status(self) -> dict:
        """Get current camera status including recording state."""
        resp = self._run_in_thread(self._gopro.http_command.get_camera_status())
        return resp.data
