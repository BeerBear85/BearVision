"""UTC-only datetime type used at BearVision system boundaries."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated

from pydantic import AfterValidator, AwareDatetime


def require_utc(value: datetime) -> datetime:
    """Reject aware datetimes that are not expressed as UTC."""

    if value.utcoffset() != timedelta(0):
        raise ValueError("datetime must be UTC")
    return value


UtcDatetime = Annotated[AwareDatetime, AfterValidator(require_utc)]
