import asyncio
from types import SimpleNamespace

import pytest
from open_gopro.models.constants import SettingId, settings

from scripts.stop_gopro_hindsight import stop_hindsight_before_redeployment


class Response:
    def __init__(self, data=None, *, ok=True) -> None:
        self.data = data
        self.ok = ok


def test_redeployment_cleanup_stops_hindsight_without_opening_usb_control() -> None:
    async def exercise() -> None:
        calls: list[object] = []
        states = [settings.Hindsight.NUM_15_SECONDS, settings.Hindsight.OFF]

        async def discover(service: str, timeout: int):
            calls.append(("discover", service, timeout))
            return SimpleNamespace(name="C3456789012345._gopro-web._tcp.local.")

        class Camera:
            def __init__(self, *, serial: str) -> None:
                calls.append(("camera", serial))
                self.http_command = SimpleNamespace(get_camera_state=self.camera_state)
                self.http_setting = SimpleNamespace(
                    hindsight=SimpleNamespace(set=self.set_hindsight)
                )

            async def camera_state(self):
                return Response({SettingId.HINDSIGHT: states.pop(0)})

            async def set_hindsight(self, value):
                calls.append(("set", value))
                return Response()

        serial = await stop_hindsight_before_redeployment(
            timeout_s=3,
            discover=discover,
            camera_factory=Camera,
        )

        assert serial == "C3456789012345"
        assert calls == [
            ("discover", "_gopro-web._tcp.local.", 3),
            ("camera", "C3456789012345"),
            ("set", settings.Hindsight.OFF),
        ]
        assert states == []

    asyncio.run(exercise())


def test_redeployment_cleanup_fails_closed_without_off_confirmation() -> None:
    async def exercise() -> None:
        async def discover(service: str, timeout: int):
            return SimpleNamespace(name="C3456789012345._gopro-web._tcp.local.")

        class Camera:
            def __init__(self, *, serial: str) -> None:
                self.http_command = SimpleNamespace(get_camera_state=self.camera_state)
                self.http_setting = SimpleNamespace(
                    hindsight=SimpleNamespace(set=self.set_hindsight)
                )

            async def camera_state(self):
                return Response(
                    {SettingId.HINDSIGHT: settings.Hindsight.NUM_15_SECONDS}
                )

            async def set_hindsight(self, value):
                return Response(ok=False)

        with pytest.raises(RuntimeError, match="did not confirm HindSight OFF"):
            await stop_hindsight_before_redeployment(
                discover=discover,
                camera_factory=Camera,
            )

    asyncio.run(exercise())
