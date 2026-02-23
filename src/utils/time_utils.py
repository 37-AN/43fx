"""Timezone and session utility helpers."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


def convert_to_timezone(ts: datetime, timezone_name: str) -> datetime:
    """Convert timezone-aware timestamp to target timezone."""
    if ts.tzinfo is None:
        raise ValueError("Timestamp must be timezone-aware.")
    return ts.astimezone(ZoneInfo(timezone_name))


def is_within_session(ts: datetime, session_config: dict) -> bool:
    """Check whether timestamp falls within configured session hours.

    Expected session_config fields:
    - start_hour: int (0-23)
    - end_hour: int (0-23)
    - timezone: str (IANA timezone)
    """
    start_hour = int(session_config.get("start_hour", 8))
    end_hour = int(session_config.get("end_hour", 17))
    timezone_name = session_config.get("timezone", "UTC")

    local_ts = convert_to_timezone(ts, timezone_name)
    hour = local_ts.hour

    if start_hour <= end_hour:
        return start_hour <= hour < end_hour

    return hour >= start_hour or hour < end_hour
