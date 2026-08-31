"""Compatibility facade for the packaged BearVision 3 GoPro integration."""

from open_gopro import WiredGoPro

from bearvision.integrations.gopro_controller import GoProController as _GoProController


class GoProController(_GoProController):
    """Keep legacy ``GoProController.WiredGoPro`` test injection working."""

    def __init__(self, target: str | None = None) -> None:
        self._gopro = WiredGoPro(target=target)
        self._loop = None
        self._loop_thread = None
        self._is_connected = False
