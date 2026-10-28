"""Mission replay clock (PLAN.md 0.2): maps dataset time to mission day N and advances as_of at a chosen speed.

`speed` is dataset seconds per real second (720 = 1 mission hour per 5 s). The clock only moves while
`running`, never leaves [start, end] (the dataset span), and is thread-safe.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from backend.timeutil import iso_z, parse_ts

DATASET_START = datetime(2015, 12, 14, tzinfo=UTC)  # mission day 1
DATASET_END = datetime(2016, 1, 18, 23, 59, 59, tzinfo=UTC)
DEFAULT_AS_OF = datetime(2016, 1, 3, 12, tzinfo=UTC)
DEFAULT_SPEED = 720.0
MAX_SPEED = 1e6


def mission_day(as_of: datetime) -> int:
    """Whole days since 2015-12-14, plus one."""
    return (as_of - DATASET_START).days + 1


class MissionClock:
    def __init__(
        self,
        as_of: datetime = DEFAULT_AS_OF,
        speed: float = DEFAULT_SPEED,
        running: bool = False,
        start: datetime = DATASET_START,
        end: datetime = DATASET_END,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.start, self.end, self._time = start, end, time_fn
        self._lock = threading.Lock()
        self._as_of = min(max(as_of, start), end)
        self._speed = float(speed)
        self._running = bool(running)
        self._tick = time_fn()

    def _advance(self) -> None:  # caller holds the lock
        t = self._time()
        if self._running:
            span = (self.end - self._as_of).total_seconds()
            try:
                delta = min(max((t - self._tick) * self._speed, 0.0), span)
                self._as_of = min(self._as_of + timedelta(seconds=delta), self.end)
            except (OverflowError, ValueError):
                self._as_of = self.end
            if self._as_of >= self.end:
                self._running = False  # the replay is over
        self._tick = t

    def now(self) -> datetime:
        with self._lock:
            self._advance()
            return self._as_of

    @property
    def running(self) -> bool:
        with self._lock:
            self._advance()
            return self._running

    @property
    def speed(self) -> float:
        with self._lock:
            return self._speed

    @property
    def mission_day(self) -> int:
        return mission_day(self.now())

    def set(
        self,
        as_of: datetime | str | None = None,
        speed: float | None = None,
        running: bool | None = None,
    ) -> None:
        with self._lock:
            self._advance()  # settle elapsed time at the old speed before changing anything
            if speed is not None and not (math.isfinite(speed) and 0 < speed <= MAX_SPEED):
                raise ValueError(f"speed must be finite and in (0, {MAX_SPEED:g}]")
            if as_of is not None:
                try:
                    parsed = parse_ts(as_of)
                except (ValueError, TypeError, OverflowError):
                    raise ValueError(f"bad as_of {as_of!r}") from None
                self._as_of = min(max(parsed, self.start), self.end)
            if speed is not None:
                self._speed = float(speed)
            if running is not None:
                self._running = bool(running) and self._as_of < self.end

    def state(self) -> dict:
        with self._lock:
            self._advance()
            return {
                "as_of": iso_z(self._as_of),
                "speed": self._speed,
                "mission_day": mission_day(self._as_of),
                "running": self._running,
            }


CLOCK = MissionClock()
