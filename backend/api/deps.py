"""Shared API plumbing: per-request DB connection, the as_of query parameter, labels."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException, Query

from backend.config import ROOT, TrayConfig, load_settings, load_trays
from backend.db import connect, init_db
from backend.sim.clock import CLOCK
from backend.timeutil import parse_ts

BADGES = {
    "control": "DATASET",
    "no_water": "SIMULATED FAULT (image transform)",
    "low_light": "SIMULATED FAULT (time-warp)",
    "humid_no_fan": "SIMULATED (sensors)",
}


_migrated: set[Path] = set()


def open_db() -> sqlite3.Connection:
    """Connection to the configured DB. The first open of each path per process runs init_db, which creates a
    missing DB and migrates an old one (added columns); later opens skip it so requests don't take the lock."""
    path = ROOT / load_settings().db_path
    if path in _migrated and path.exists():
        return connect(path)
    conn = init_db(path)
    _migrated.add(path)
    return conn


def get_conn() -> Iterator[sqlite3.Connection]:
    conn = open_db()
    try:
        yield conn
    finally:
        conn.close()


def get_as_of(as_of: str | None = Query(None, description="ISO UTC; default: the mission clock")) -> datetime:
    if as_of is None:
        return CLOCK.now()
    try:
        return min(max(parse_ts(as_of), CLOCK.start), CLOCK.end)
    except (ValueError, TypeError, OverflowError):
        raise HTTPException(422, f"bad as_of {as_of!r}") from None


def tray_cfg(tray_id: str) -> TrayConfig | None:
    return next((t for t in load_trays().trays if t.id == tray_id), None)


def require_tray(tray_id: str) -> TrayConfig:
    t = tray_cfg(tray_id)
    if t is None:
        raise HTTPException(404, f"unknown tray {tray_id!r}")
    return t
