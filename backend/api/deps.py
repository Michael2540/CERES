"""Shared API plumbing: per-request DB connection, the as_of query parameter, labels."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import datetime

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


def open_db() -> sqlite3.Connection:
    path = ROOT / load_settings().db_path
    return connect(path) if path.is_file() else init_db(path)  # a missing DB gets the schema


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
