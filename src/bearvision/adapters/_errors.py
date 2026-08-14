"""Translate implementation exceptions into stable port failures."""

from bearvision.ports import (
    ComponentError,
    ComponentTimeout,
    ComponentUnavailable,
    InvalidComponentData,
    PermanentComponentError,
)


def translated_error(exc: Exception, operation: str) -> ComponentError:
    if isinstance(exc, ComponentError):
        return exc
    message = f"{operation}: {exc}"
    if isinstance(exc, TimeoutError):
        return ComponentTimeout(message)
    if isinstance(exc, (ConnectionError, OSError)):
        return ComponentUnavailable(message)
    status_code = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status_code in {408, 409, 429} or (
        isinstance(status_code, int) and status_code >= 500
    ):
        return ComponentUnavailable(message)
    if status_code in {400, 401, 403, 404, 422}:
        return PermanentComponentError(message)
    if isinstance(exc, (KeyError, TypeError, ValueError)):
        return InvalidComponentData(message)
    return ComponentError(message)
