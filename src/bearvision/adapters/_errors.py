"""Translate implementation exceptions into stable port failures."""

from bearvision.ports import (
    ComponentError,
    ComponentTimeout,
    ComponentUnavailable,
    InvalidComponentData,
)


def translated_error(exc: Exception, operation: str) -> ComponentError:
    if isinstance(exc, ComponentError):
        return exc
    message = f"{operation}: {exc}"
    if isinstance(exc, TimeoutError):
        return ComponentTimeout(message)
    if isinstance(exc, (ConnectionError, OSError)):
        return ComponentUnavailable(message)
    if isinstance(exc, (KeyError, TypeError, ValueError)):
        return InvalidComponentData(message)
    return ComponentError(message)
