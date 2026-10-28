"""GET /trays and /trays/{id}: status, latest metrics and open alerts as of the mission clock."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends

from backend.api import deps
from backend.api.alerts import visible_alerts
from backend.api.series import canopy_table, climate_owner, growth_points
from backend.config import TrayConfig, load_trays
from backend.science.vpd import vpd_kpa
from backend.sim.clock import DATASET_START, mission_day
from backend.timeutil import iso_z

router = APIRouter()

STATUS_BY_SEVERITY = {"alert": 2, "warning": 1, "watch": 1}


def open_alerts(conn: sqlite3.Connection, tray_id: str, as_of: datetime) -> list[dict]:
    """Episodes of this tray that fired within the last 2 h before as_of (see api/alerts.py)."""
    return [
        {"id": a["id"], "rule": a["rule"], "severity": a["severity"]}
        for a in visible_alerts(conn, as_of, tray_id)
        if a["status"] == "open"
    ]


def shared_alerts(conn: sqlite3.Connection, as_of: datetime) -> list[dict]:
    """Open alerts of the shared air sensors (tray_id `air`), same shape as GET /alerts items."""
    return [a for a in visible_alerts(conn, as_of, "air") if a["status"] == "open"]


def status_of(alerts: list[dict]) -> str:
    worst = max((STATUS_BY_SEVERITY.get(a["severity"], 1) for a in alerts), default=0)
    return ("nominal", "watch", "alert")[worst]


def latest_frame_id(conn: sqlite3.Connection, tray_id: str, as_of: str) -> str | None:
    r = conn.execute(
        "SELECT id FROM frames WHERE tray_id = ? AND ts <= ? ORDER BY ts DESC LIMIT 1", (tray_id, as_of)
    ).fetchone()
    return r["id"] if r else None


def summary(
    conn: sqlite3.Connection, t: TrayConfig, as_of: datetime, shared: list[dict] | None = None
) -> dict:
    hi = iso_z(as_of)
    alerts = open_alerts(conn, t.id, as_of)
    return {
        "id": t.id,
        "crop": t.crop,
        "treatment": t.treatment,
        "status": status_of(alerts),
        "fault_start": t.fault_start,
        "reference_tray_id": t.reference_tray_id,
        "badge": deps.BADGES[t.treatment],
        "open_alerts": len(alerts),
        "latest_frame_id": latest_frame_id(conn, t.id, hi),
        "shared_alerts": shared if shared is not None else shared_alerts(conn, as_of),
    }


@router.get("/trays")
def list_trays(
    as_of: datetime = Depends(deps.get_as_of), conn: sqlite3.Connection = Depends(deps.get_conn)
) -> dict:
    shared = shared_alerts(conn, as_of)
    return {"items": [summary(conn, t, as_of, shared) for t in load_trays().trays]}


def _latest_value(conn: sqlite3.Connection, owner: str, metric: str, hi: str) -> dict | None:
    r = conn.execute(
        "SELECT ts, value FROM sensor_readings WHERE tray_id = ? AND metric = ? AND ts <= ?"
        " ORDER BY ts DESC LIMIT 1",
        (owner, metric, hi),
    ).fetchone()
    return dict(r) if r else None


@router.get("/trays/{tray_id}")
def tray_detail(
    tray_id: str,
    as_of: datetime = Depends(deps.get_as_of),
    conn: sqlite3.Connection = Depends(deps.get_conn),
) -> dict:
    t = deps.require_tray(tray_id)
    hi = iso_z(as_of)
    alerts = open_alerts(conn, t.id, as_of)
    cal = canopy_table(conn, t.id, iso_z(as_of - timedelta(hours=48)), hi)
    last = cal[-1] if cal else None
    owner = climate_owner(conn, t.id)
    temp, rh = _latest_value(conn, owner, "temp_c", hi), _latest_value(conn, owner, "rh_pct", hi)
    soil = _latest_value(conn, t.id, "soil_vwc", hi)
    vpd = vpd_kpa(temp["value"], rh["value"]) if temp and rh and temp["ts"] == rh["ts"] else None
    latest = {
        "ts": last["ts"] if last else None,
        "frame_id": last["frame_id"] if last else None,
        "area_px": last["area_px"] if last else None,
        "area_cm2": last["area_cm2"] if last else None,
        "growth_rate": growth_points(cal)[last["ts"]] if last else None,
        "exg": last["exg"] if last else None,
        "chlorosis_frac": last["chlorosis_frac"] if last else None,
        "drift_z": last["drift_z"] if last else None,
        "soil_vwc": soil["value"] if soil else None,
        "temp_c": temp["value"] if temp else None,
        "rh_pct": rh["value"] if rh else None,
        "vpd_kpa": vpd,
    }
    s = summary(conn, t, as_of)
    return {
        "id": t.id,
        "crop": t.crop,
        "treatment": t.treatment,
        "status": s["status"],
        "fault_start": t.fault_start,
        "reference_tray_id": t.reference_tray_id,
        "age_days": (as_of - DATASET_START).days,
        "mission_day": mission_day(as_of),
        "badge": s["badge"],
        "calib_flag": t.calib_flag,
        "latest": latest,
        "open_alerts": alerts,
        "shared_alerts": s["shared_alerts"],
    }
