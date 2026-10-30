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
    """True when the lamp schedule (UTC HH:MM on/off) has the lamp on at dt. An overnight schedule (off before
    on, e.g. 20:00-06:00) is on from `on` to midnight and from midnight to `off`; on == off means always off."""
    t = dt.astimezone(UTC)
    minute, on, off = t.hour * 60 + t.minute, _minutes(lamp.on), _minutes(lamp.off)
    return on <= minute < off if on <= off else minute >= on or minute < off
