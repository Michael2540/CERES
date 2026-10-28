"""GET /alerts: alert episodes as of as_of.

An `alerts` row is an episode (first_ts..last_ts) and `alert_events` has one row per cycle in which the rule
fired. At as_of an episode is visible iff first_ts <= as_of; its ts, evidence_ids and detail come from its
latest event <= as_of, and it is open iff it has an event in (as_of - 2 h, as_of]. Nothing after as_of leaks.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends

from backend.api import deps
from backend.science.rules import episode_gap_h
from backend.timeutil import iso_z

router = APIRouter()


def visible_alerts(
    conn: sqlite3.Connection, as_of: datetime, tray_id: str | None = None, since: str | None = None
) -> list[dict]:
    hi = iso_z(as_of)
    q = (
        "SELECT a.id, a.tray_id, a.rule, a.severity, a.threshold_tag, e.ts, e.evidence_ids, e.detail,"
        " (SELECT s.id FROM assessments s WHERE s.alert_id = a.id AND s.ts <= :hi ORDER BY s.ts DESC LIMIT 1)"
        " AS assessment_id FROM alerts a JOIN alert_events e ON e.alert_id = a.id AND e.ts = ("
        " SELECT MAX(x.ts) FROM alert_events x WHERE x.alert_id = a.id AND x.ts <= :hi)"
        " WHERE a.first_ts <= :hi"
    )
    args: dict = {"hi": hi}
    if tray_id:
        q += " AND a.tray_id = :tray"
        args["tray"] = tray_id
    if since:
        q += " AND e.ts >= :since"
        args["since"] = since
    return [
        {
            "id": r["id"],
            "ts": r["ts"],
            "tray_id": r["tray_id"],
            "rule": r["rule"],
            "severity": r["severity"],
            "evidence_ids": json.loads(r["evidence_ids"] or "[]"),
            "threshold_tag": r["threshold_tag"],
            "assessment_id": r["assessment_id"],
            "detail": r["detail"],
            "status": "open" if r["ts"] > iso_z(as_of - timedelta(hours=episode_gap_h(r["rule"]))) else "resolved",
        }
        for r in conn.execute(q + " ORDER BY e.ts DESC, a.id", args)
    ]


@router.get("/alerts")
def list_alerts(
    tray_id: str | None = None,
    since: str | None = None,
    as_of: datetime = Depends(deps.get_as_of),
    conn: sqlite3.Connection = Depends(deps.get_conn),
) -> dict:
    return {"items": visible_alerts(conn, as_of, tray_id, since)}
