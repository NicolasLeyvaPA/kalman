"""Time parsing and arithmetic helpers shared across services.

Every datetime in the system is timezone-aware (UTC). Convert at the
boundary, store TZ-aware, never compare naive vs aware datetimes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def to_utc(value: Any) -> datetime | None:
    """Parse heterogeneous input into a timezone-aware UTC datetime.

    Accepts: ``datetime`` (naive treated as UTC), ``int``/``float`` Unix
    seconds, ISO-8601 strings (with or without trailing ``Z``), numeric
    strings (treated as Unix seconds). Returns ``None`` for anything else.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            try:
                return datetime.fromtimestamp(float(s), tz=UTC)
            except (TypeError, ValueError, OverflowError, OSError):
                return None
    return None


def utc_now() -> datetime:
    return datetime.now(tz=UTC)
