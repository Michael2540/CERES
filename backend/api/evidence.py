"""GET /evidence/{id}: resolve any evidence id (frame, reading, canopy/derived metric, alert, note).

Id forms (API.md): frame `<tray>:<ts>`, reading and metric `<tray>:<metric>:<ts>` (tray may be `air`),
alert `alert_<...>`, crew note by its row id. Nothing after as_of is resolvable.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException

from backend.api import deps
from backend.api.alerts import visible_alerts
from backend.api.series import ALL_METRICS, CANOPY_METRICS, canopy_table, growth_points, vpd_points
from backend.timeutil import iso_z, parse_ts

router = APIRouter()
_METRIC_ID = re.compile(r"^(?P<tray>[A-Za-z0-9]+):(?P<metric>[a-z0-9_]+):(?P<ts>\d{4}-\d\d-\d\dT[\d:]+Z)$")


def resolve(conn: sqlite3.Connection, eid: str, as_of: datetime) -> dict | None:
    hi = iso_z(as_of)
    f = conn.execute("SELECT * FROM frames WHERE id = ?", (eid,)).fetchone()
    if f and f["ts"] <= hi:
        return {
            "id": eid, "kind": "frame", "tray_id": f["tray_id"], "ts": f["ts"], "frame_id": eid,
            "source": f["source"],
        }
    a = next((x for x in visible_alerts(conn, as_of) if x["id"] == eid), None)
    if a:
        return {"id": eid, "kind": "alert", "tray_id": a["tray_id"], "ts": a["ts"]}
    r = conn.execute("SELECT * FROM sensor_readings WHERE id = ?", (eid,)).fetchone()
    if r and r["ts"] <= hi:
        return {
            "id": eid, "kind": "reading", "tray_id": r["tray_id"], "ts": r["ts"], "metric": r["metric"],
            "value": r["value"], "source": r["source"],
        }
    n = conn.execute("SELECT id, ts FROM crew_notes WHERE id = ?", (eid,)).fetchone()
    if n and n["ts"] <= hi:
        return {"id": eid, "kind": "note", "tray_id": None, "ts": n["ts"]}
    m = _METRIC_ID.match(eid)
    if not m or m["ts"] > hi or m["metric"] not in ALL_METRICS:
        return None
    tray, metric, ts = m["tray"], m["metric"], m["ts"]
    frame_id, source = None, "derived"
    if metric == "vpd_kpa":
        pts = vpd_points(conn, tray, iso_z(parse_ts(ts) - timedelta(seconds=1)), ts)
        value = pts[0]["value"] if pts else None
    elif metric in CANOPY_METRICS:
        rows = canopy_table(conn, tray, iso_z(parse_ts(ts) - timedelta(hours=24)), ts)
        row = next((x for x in rows if x["ts"] == ts), None)
        if row is None:
            return None
        value = growth_points(rows)[ts] if metric == "growth_rate" else row[metric]
        frame_id, source = row["frame_id"], row["source"]  # the frame's: dataset|fault_transform|fault_timewarp
    else:
        return None
    if value is None:
        return None
    out = {
        "id": eid, "kind": "metric", "tray_id": tray, "ts": ts, "metric": metric, "value": value,
        "source": source,
    }
    if frame_id:
        out["frame_id"] = frame_id
    return out


@router.get("/evidence/{evidence_id:path}")
def evidence(
    evidence_id: str,
    as_of: datetime = Depends(deps.get_as_of),
    conn: sqlite3.Connection = Depends(deps.get_conn),
) -> dict:
    out = resolve(conn, evidence_id, as_of)
    if out is None:
        raise HTTPException(404, f"unknown or future evidence id {evidence_id!r}")
    return out
