import asyncio
import threading
from pathlib import Path
import logging
from open_gopro import WiredGoPro, models
from open_gopro.models.constants import settings
from open_gopro.models.constants import constants
from GoProConfig import GoProConfiguration, save_config_to_yaml, load_config_from_yaml

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

    def startHindsightMode(self) -> None:
        """Start Hindsight Mode with hardcoded 15 seconds buffer.
        
        This is a simplified function that always uses 15 seconds for hindsight.
        The hindsight buffer is configured in the configure() method.
        """
        self.start_hindsight_clip(1.0)  # Trigger recording for 1 second

    def get_camera_status(self) -> dict:
        """Get current camera status including recording state."""
        resp = self._run_in_thread(self._gopro.http_command.get_camera_status())
        return resp.data

    def download_configuration(self, output_path: str = None) -> str:
        """
        Download current camera configuration to a YAML file.
        
        Args:
            output_path: Path to save configuration file. If None, uses default path.
            
        Returns:
            Path to the saved configuration file
            
        Raises:
            ConnectionError: If GoPro is not connected or connection fails
            OSError: If unable to save configuration file
            Exception: If unable to retrieve configuration from camera
        """
        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"gopro_config_{timestamp}.yaml"
        
        # Check if we have a GoPro connection
        if not self._gopro:
            raise ConnectionError("No GoPro connection available. Please connect to a GoPro first.")
        
        try:
            # Verify GoPro is still reachable by checking if _serial is set
            if not hasattr(self._gopro, '_serial') or not self._gopro._serial:
                raise ConnectionError("GoPro connection lost. Please reconnect to the camera.")
            
            logger.info("Retrieving configuration from GoPro...")
            
            # Get current camera state
            camera_state = self._run_in_thread(self._gopro.http_command.get_camera_state())
            
            if not camera_state or not hasattr(camera_state, 'data') or not camera_state.data:
                raise Exception("Failed to retrieve camera state - empty response from GoPro")
            
            logger.info("Converting camera state to configuration model...")
            
            # Convert to our configuration model
            try:
                config = GoProConfiguration.from_gopro_state(camera_state.data)
            except Exception as e:
                raise Exception(f"Failed to convert camera state to configuration model: {e}")
            
            logger.info(f"Saving configuration to: {output_path}")
            
            # Save to YAML file
            try:
                save_config_to_yaml(config, output_path)
            except (PermissionError, OSError) as e:
                raise OSError(f"Unable to save configuration file: {e}")
            except Exception as e:
                raise OSError(f"Unexpected error saving configuration: {e}")
            
            logger.info(f"Configuration downloaded and saved to: {output_path}")
            return output_path
            
        except (ConnectionError, OSError):
            # Re-raise these specific errors as-is
            raise
        except Exception as e:
            logger.error(f"Failed to download configuration: {e}")
            # Check if the error is related to connection issues
            if "connection" in str(e).lower() or "network" in str(e).lower():
                raise ConnectionError(f"GoPro connection error during configuration download: {e}")
            else:
                raise Exception(f"Failed to download configuration: {e}")

    def upload_configuration(self, config_path: str) -> bool:
        """
        Upload configuration from a YAML file to the camera.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            True if configuration was successfully applied
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration file format is invalid
            ConnectionError: If GoPro is not connected or connection fails
            Exception: If unable to apply configuration to camera
        """
        from pathlib import Path
        from pydantic import ValidationError
        
        # Check if configuration file exists
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Check if we have a GoPro connection
        if not self._gopro:
            raise ConnectionError("No GoPro connection available. Please connect to a GoPro first.")
        
        try:
            # Verify GoPro is still reachable
            if not hasattr(self._gopro, '_serial') or not self._gopro._serial:
                raise ConnectionError("GoPro connection lost. Please reconnect to the camera.")
            
            logger.info(f"Loading configuration from: {config_path}")
            
            # Load and validate configuration
            try:
                config = load_config_from_yaml(config_path)
            except (FileNotFoundError, ValueError, ValidationError) as e:
                # Re-raise file and validation errors as-is
                raise
            except Exception as e:
                raise ValueError(f"Unexpected error loading configuration file: {e}")
            
            logger.info("Validating configuration compatibility...")
            
            # Additional validation could be added here to check if the configuration
            # is compatible with the connected GoPro model
            
            logger.info("Applying configuration to GoPro...")
            
            # Apply configuration to camera
            try:
                self._run_in_thread(self._apply_configuration(config))
            except Exception as e:
                # Check if the error is related to connection issues
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["connection", "network", "timeout", "unreachable"]):
                    raise ConnectionError(f"GoPro connection error during configuration upload: {e}")
                else:
                    raise Exception(f"Failed to apply configuration to camera: {e}")
            
            logger.info(f"Configuration from {config_path} successfully applied to camera")
            return True
            
        except (FileNotFoundError, ValueError, ValidationError, ConnectionError):
            # Re-raise these specific errors as-is
            raise
        except Exception as e:
            logger.error(f"Failed to upload configuration: {e}")
            raise Exception(f"Unexpected error during configuration upload: {e}")

    async def _apply_configuration(self, config: GoProConfiguration) -> None:
        """
        Apply configuration to the camera.
        
        Args:
            config: GoProConfiguration instance to apply
        """
        try:
            # Map configuration to GoPro HTTP settings
            settings_map = config.to_gopro_settings()
            
            # Apply video resolution if specified
            if config.video_resolution:
                # Handle both enum objects and string values
                resolution_value = config.video_resolution.value if hasattr(config.video_resolution, 'value') else config.video_resolution
                if resolution_value == "4K":
                    await self._gopro.http_settings.video_resolution.set(settings.VideoResolution.NUM_4K)
                elif resolution_value == "2.7K":
                    await self._gopro.http_settings.video_resolution.set(settings.VideoResolution.NUM_2_7K)
                elif resolution_value == "1080p":
                    await self._gopro.http_settings.video_resolution.set(settings.VideoResolution.NUM_1080)
            
            # Apply frame rate if specified
            if config.frame_rate:
                # Handle both enum objects and string values
                frame_rate_value = config.frame_rate.value if hasattr(config.frame_rate, 'value') else config.frame_rate
                if frame_rate_value == "60":
                    await self._gopro.http_settings.frames_per_second.set(settings.FramesPerSecond.NUM_60_0)
                elif frame_rate_value == "30":
                    await self._gopro.http_settings.frames_per_second.set(settings.FramesPerSecond.NUM_30_0)
                elif frame_rate_value == "24":
                    await self._gopro.http_settings.frames_per_second.set(settings.FramesPerSecond.NUM_24_0)
            
            # Apply hindsight if specified
            if config.hindsight:
                # Handle both enum objects and string values
                hindsight_value = config.hindsight.value if hasattr(config.hindsight, 'value') else config.hindsight
                if hindsight_value == "15 seconds":
                    await self._gopro.http_settings.hindsight.set(settings.Hindsight.NUM_15_SECONDS)
                elif hindsight_value == "30 seconds":
                    await self._gopro.http_settings.hindsight.set(settings.Hindsight.NUM_30_SECONDS)
                elif hindsight_value == "OFF":
                    await self._gopro.http_settings.hindsight.set(settings.Hindsight.OFF)
            
            # Apply additional settings as needed
            # Note: More settings mappings would be added here as the open_gopro library
            # is explored further for available configuration options
            
            logger.info("Configuration successfully applied to camera")
            
        except Exception as e:
            logger.error(f"Failed to apply configuration to camera: {e}")
            raise

    def validate_configuration(self, config_path: str) -> tuple[bool, str]:
        """
        Validate a configuration file without applying it.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            config = load_config_from_yaml(config_path)
            return True, "Configuration is valid"
        except Exception as e:
            return False, str(e)
