"""Stable component failure categories used by core orchestration."""


class ComponentError(RuntimeError):
    """Base error raised by a component adapter."""


class ComponentUnavailable(ComponentError):
    """The component cannot currently accept work."""


class ComponentTimeout(ComponentError):
    """The operation exceeded its monotonic deadline."""


class InvalidComponentData(ComponentError):
    """The component returned data that violates its contract."""


class PermanentComponentError(ComponentError):
    """Retrying without configuration or hardware changes cannot succeed."""
