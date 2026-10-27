"""Small UTC time helpers shared by the simulator, the sensor reader and the rules."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol


class _Lamp(Protocol):
    on: str
    off: str


def parse_ts(s: str | datetime) -> datetime:
    """Parse an ISO-8601 timestamp (a trailing Z means UTC); naive values are taken as UTC."""
    if isinstance(s, datetime):
        dt = s
    elif s.endswith("Z"):  # fast path for the canonical form used everywhere
        return datetime.fromisoformat(s[:-1]).replace(tzinfo=UTC)
    else:
        dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def iso_z(dt: datetime) -> str:
    """Format as ISO-8601 UTC with a trailing Z and whole seconds."""
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def lamp_is_on(lamp: _Lamp, dt: datetime) -> bool:
    """True when the lamp schedule (UTC HH:MM on/off, same day) has the lamp on at dt."""
    minute = dt.astimezone(UTC).hour * 60 + dt.astimezone(UTC).minute
    return _minutes(lamp.on) <= minute < _minutes(lamp.off)
