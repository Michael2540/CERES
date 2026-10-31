"""Earth-Mars link state machine (M8).

States: `up`, `delayed(delay_s)`, `down`. While the state is not `up`, Earth-bound agent calls are queued
(backend/link/queue.py) instead of executed. The delay is SIMULATED: a demo uses a scaled delay in wall-clock seconds
(the real one-way light delay is 3-22 minutes, see LIGHT_DELAY_NOTE, which is text only and never used in a computation).

Persistence: every change is a row of `link_events` (state, delay_s, mission time `since`, wall time `since_wall`); the
last row is the current state, so it survives a restart. With no row the link is `up`.

Transitions are explicit (TRANSITIONS); a transition to the current state is refused, except `delayed -> delayed` with a
different delay. `deliverable(...)` is the single rule for when a queued item may leave: up -> always, down -> never,
delayed -> `delay_s` wall-clock seconds after the item was queued (with the CURRENT delay).
"""

from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime

from backend.sim.clock import DATASET_START
from backend.timeutil import iso_z

STATES = ("up", "delayed", "down")
TRANSITIONS: dict[str, tuple[str, ...]] = {
    "up": ("delayed", "down"),
    "delayed": ("up", "down", "delayed"),  # delayed -> delayed only changes the delay
    "down": ("up", "delayed"),
}
MAX_DELAY_S = 3600.0
LIGHT_DELAY_NOTE = (
    "Simulated link. The real one-way light delay between Earth and Mars is 3-22 min; the demo uses a scaled delay in "
    "wall-clock seconds."
)


class InvalidTransition(ValueError):
    """The requested state change is not allowed (409 in the API) or its arguments are bad (422)."""


@dataclass(frozen=True)
class LinkState:
    state: str = "up"
    delay_s: float = 0.0
    since: str = iso_z(DATASET_START)  # mission time of the change
    since_wall: float = 0.0  # wall-clock unix time of the change


def validate_args(state: str, delay_s: float | None) -> float:
    """The delay to store for `state`: 0 for up/down; for delayed a finite number in (0, MAX_DELAY_S]."""
    if state not in STATES:
        raise ValueError(f"state must be one of {STATES}, got {state!r}")
    if state != "delayed":
        if delay_s not in (None, 0):
            raise ValueError(f"delay_s is only for state 'delayed' (got {delay_s!r} with {state!r})")
        return 0.0
    if delay_s is None or not math.isfinite(delay_s) or not 0 < delay_s <= MAX_DELAY_S:
        raise ValueError(f"state 'delayed' needs delay_s in (0, {MAX_DELAY_S:g}]")
    return float(delay_s)


def check_transition(cur: LinkState, state: str, delay_s: float) -> None:
    if state not in TRANSITIONS[cur.state]:
        raise InvalidTransition(f"link is already {cur.state}" + (f" ({cur.delay_s:g} s)" if cur.state == "delayed" else ""))
    if state == "delayed" and cur.state == "delayed" and delay_s == cur.delay_s:
        raise InvalidTransition(f"link is already delayed by {cur.delay_s:g} s")


def _row(r: sqlite3.Row) -> LinkState:
    return LinkState(r["state"], float(r["delay_s"] or 0.0), r["since"], float(r["since_wall"] or 0.0))


def current(conn: sqlite3.Connection) -> LinkState:
    r = conn.execute("SELECT * FROM link_events ORDER BY id DESC LIMIT 1").fetchone()
    return _row(r) if r else LinkState()


def state_at(conn: sqlite3.Connection, as_of: datetime) -> LinkState:
    """The state as of mission time `as_of` (the last change with since <= as_of), for time-gated reads."""
    r = conn.execute("SELECT * FROM link_events WHERE since <= ? ORDER BY id DESC LIMIT 1", (iso_z(as_of),)).fetchone()
    return _row(r) if r else LinkState()


def transition(
    conn: sqlite3.Connection, state: str, delay_s: float | None, as_of: datetime, now: float | None = None
) -> LinkState:
    """Change the link state (persisted). Raises ValueError for bad arguments, InvalidTransition for a refused change."""
    d = validate_args(state, delay_s)
    cur = current(conn)
    check_transition(cur, state, d)
    wall = time.time() if now is None else now
    conn.execute(
        "INSERT INTO link_events(state, delay_s, since, since_wall) VALUES (?,?,?,?)", (state, d, iso_z(as_of), wall)
    )
    conn.commit()
    return current(conn)


def deliverable(link: LinkState, queued_at: float, now: float) -> bool:
    """May an item queued at wall time `queued_at` be delivered to Earth at wall time `now`?"""
    if link.state == "up":
        return True
    if link.state == "down":
        return False
    return now >= queued_at + link.delay_s


def seconds_until(link: LinkState, queued_at: float, now: float) -> float | None:
    """Wall seconds until the item is deliverable (0 when it is now, None while the link is down)."""
    if link.state == "up":
        return 0.0
    if link.state == "down":
        return None
    return max(0.0, queued_at + link.delay_s - now)
