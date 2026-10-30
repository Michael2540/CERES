"""GET /trays/{id}/series: sensor metrics, canopy metrics and on-the-fly VPD, all bounded by as_of.

Point ids are `<tray>:<metric>:<ts>` for every metric, so GET /evidence/{id} can resolve any of them.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api import deps
from backend.perception import pots
from backend.perception.metrics import growth_rate_24h
from backend.science.vpd import vpd_kpa
from backend.timeutil import iso_z, parse_ts

router = APIRouter()

SENSOR_METRICS = ("temp_c", "rh_pct", "co2_ppm", "soil_raw", "soil_vwc", "lamp_on", "ppfd_est")
CANOPY_METRICS = ("area_px", "area_cm2", "exg", "chlorosis_frac", "growth_rate", "drift_z", "plant_count")
ALL_METRICS = (*SENSOR_METRICS, *CANOPY_METRICS, "vpd_kpa")
UNITS = {
    "temp_c": "C",
    "rh_pct": "%",
    "co2_ppm": "ppm",
    "soil_raw": "raw",
    "soil_vwc": "m3/m3",
    "lamp_on": "0/1",
    "ppfd_est": "umol/m2/s",
    "area_px": "px",
    "area_cm2": "cm2 (nominal)",
    "exg": "index",
    "chlorosis_frac": "fraction",
    "growth_rate": "ln(area)/day",
    "drift_z": "z",
    "plant_count": "plants",
    "vpd_kpa": "kPa",
}


CLIMATE_METRICS = ("temp_c", "rh_pct", "co2_ppm", "vpd_kpa")


def point(tray_id: str, metric: str, ts: str, value: float, source: str) -> dict:
    return {"id": f"{tray_id}:{metric}:{ts}", "ts": ts, "value": value, "source": source}


def sensor_points(conn: sqlite3.Connection, tray_id: str, metric: str, lo: str, hi: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, ts, value, source FROM sensor_readings WHERE tray_id = ? AND metric = ? AND ts > ?"
        " AND ts <= ? ORDER BY ts",
        (tray_id, metric, lo, hi),
    ).fetchall()
    return [dict(r) for r in rows]


def canopy_table(conn: sqlite3.Connection, tray_id: str, lo: str, hi: str) -> list[dict]:
    """Canopy rows with lo <= ts <= hi (oldest first) incl. drift_z when an embedding exists."""
    rows = conn.execute(
        "SELECT f.id AS frame_id, f.ts, f.source, m.area_px, m.area_cm2, m.exg, m.chlorosis_frac, m.pots_json, e.drift_z"
        " FROM masks m JOIN frames f ON f.id = m.frame_id LEFT JOIN embeddings e ON e.mask_id = m.id"
        " WHERE m.kind = 'canopy' AND f.tray_id = ? AND f.ts >= ? AND f.ts <= ? ORDER BY f.ts",
        (tray_id, lo, hi),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["pots"] = pots.load_pots(d.pop("pots_json"))
        d["plant_count"] = pots.plant_count(d["pots"])  # pots holding a plant; None without per-pot data
        out.append(d)
    return out


def growth_points(rows: list[dict]) -> dict[str, float | None]:
    """growth_rate_24h at each row's ts, from the rows (which must include the preceding 24 h)."""
    ts = [parse_ts(r["ts"]) for r in rows]
    area = [r["area_px"] for r in rows]
    per_pot = [r["pots"] for r in rows] if rows and all(r["pots"] is not None for r in rows) else None
    return {r["ts"]: growth_rate_24h(ts, area, t, per_pot) for r, t in zip(rows, ts, strict=True)}


def canopy_points(conn: sqlite3.Connection, tray_id: str, metric: str, lo: datetime, hi: datetime) -> list[dict]:
    lo_s, hi_s = iso_z(lo), iso_z(hi)
    rows = canopy_table(conn, tray_id, iso_z(lo - timedelta(hours=24)) if metric == "growth_rate" else lo_s, hi_s)
    if metric == "growth_rate":
        g = growth_points(rows)
        vals = {r["ts"]: g[r["ts"]] for r in rows}
    else:
        vals = {r["ts"]: r[metric] for r in rows}
    src = {r["ts"]: r["source"] for r in rows}  # the frame's provenance: dataset|fault_transform|fault_timewarp
    return [
        point(tray_id, metric, ts, v, src[ts]) for ts, v in vals.items() if ts > lo_s and v is not None
    ]


def climate_owner(conn: sqlite3.Connection, tray_id: str) -> str:
    """Trays with their own RH sensor (tray L under its dome) use it; everything else uses the shared air."""
    if tray_id == "air":
        return "air"
    has = conn.execute(
        "SELECT 1 FROM sensor_readings WHERE tray_id = ? AND metric = 'rh_pct' LIMIT 1", (tray_id,)
    ).fetchone()
    return tray_id if has else "air"


def metric_owner(conn: sqlite3.Connection, tray_id: str, metric: str) -> str:
    """Whose readings back a sensor/VPD series: the tray's own if it has any, else (climate metrics) `air`."""
    if metric == "vpd_kpa":
        return climate_owner(conn, tray_id)
    if tray_id == "air" or metric not in CLIMATE_METRICS:
        return tray_id
    has = conn.execute(
        "SELECT 1 FROM sensor_readings WHERE tray_id = ? AND metric = ? LIMIT 1", (tray_id, metric)
    ).fetchone()
    return tray_id if has else "air"


def vpd_points(conn: sqlite3.Connection, tray_id: str, lo: str, hi: str) -> list[dict]:
    """VPD from temp_c + rh_pct readings with identical ts (derived)."""
    owner = climate_owner(conn, tray_id)
    rows = conn.execute(
        "SELECT r.ts, t.value AS temp, r.value AS rh FROM sensor_readings r JOIN sensor_readings t"
        " ON t.tray_id = r.tray_id AND t.metric = 'temp_c' AND t.ts = r.ts"
        " WHERE r.tray_id = ? AND r.metric = 'rh_pct' AND r.ts > ? AND r.ts <= ? ORDER BY r.ts",
        (owner, lo, hi),
    ).fetchall()
    return [point(tray_id, "vpd_kpa", r["ts"], vpd_kpa(r["temp"], r["rh"]), "derived") for r in rows]


def metric_points(
    conn: sqlite3.Connection, tray_id: str, metric: str, lo: datetime, hi: datetime
) -> list[dict]:
    """Points with lo < ts <= hi."""
    if metric in SENSOR_METRICS:
        tray_id = metric_owner(conn, tray_id, metric)  # no sensor of its own: the shared air sensor
        return sensor_points(conn, tray_id, metric, iso_z(lo), iso_z(hi))
    if metric == "vpd_kpa":
        return vpd_points(conn, tray_id, iso_z(lo), iso_z(hi))
    return canopy_points(conn, tray_id, metric, lo, hi)


@router.get("/trays/{tray_id}/series")
def series(
    tray_id: str,
    metric: str,
    hours: float = Query(72, gt=0, le=24 * 60),
    as_of: datetime = Depends(deps.get_as_of),
    conn: sqlite3.Connection = Depends(deps.get_conn),
) -> dict:
    if tray_id != "air":
        deps.require_tray(tray_id)
    if metric not in ALL_METRICS:
        raise HTTPException(422, f"unknown metric {metric!r}; one of {', '.join(ALL_METRICS)}")
    items = metric_points(conn, tray_id, metric, as_of - timedelta(hours=hours), as_of)
    out = {"tray_id": tray_id, "metric": metric, "unit": UNITS[metric], "items": items}
    if tray_id != "air" and metric in CLIMATE_METRICS and metric_owner(conn, tray_id, metric) == "air":
        out["sensor_tray"] = "air"  # no sensor of its own: the shared air sensor
    return out
