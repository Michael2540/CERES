"""link_queue: Earth-bound work held while the link is not up (M8).

One item per assessed alert: kind "assess", payload {alert_id, as_of} (as_of = the mission time it was queued; the edge
triage and the Earth-side run both use it). Lifecycle (`status`):
    queued -> edge_done | edge_invalid -> sent (Earth-side run started) -> audited | failed
`edge_done` means the on-board triage ran AND passed validation. A triage that ran but failed validation (schema,
citations, numeric grounding, unbacked crew claim) is stored with its errors and gets `edge_invalid`, never edge_done.
An item whose edge triage could not run (no recording, budget) stays `queued` with `error` set and can still be drained;
an `edge_invalid` item is drained too, but its audit is "not meaningful" (agree = null).
Drain order is `ts` (mission time queued), then wall time queued, then id. Wall-clock `queued_at` drives the delay.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from backend.link import delay_sim
from backend.timeutil import iso_z

KIND = "assess"
PERCEPTION_NOTE = "frames segmented before the link dropped"  # Mars mode never re-runs perception
DRAINABLE = ("queued", "edge_done", "edge_invalid")
OPEN_STATUSES = (*DRAINABLE, "sent")


class AlreadyQueued(ValueError):
    pass


def item_id(alert_id: str, as_of: datetime) -> str:
    return f"lq_{alert_id.removeprefix('alert_')}_{iso_z(as_of).replace('-', '').replace(':', '')}"


def enqueue(conn: sqlite3.Connection, alert: dict, as_of: datetime, now: float) -> str:
    """Queue an alert for Earth. Raises AlreadyQueued if a not-failed item of the alert exists."""
    dup = conn.execute(
        "SELECT id FROM link_queue WHERE json_extract(payload, '$.alert_id') = ? AND status != 'failed'", (alert["id"],)
    ).fetchone()
    if dup:
        raise AlreadyQueued(f"alert {alert['id']} is already in the link queue as {dup['id']}")
    iid = item_id(alert["id"], as_of)
    conn.execute(
        "INSERT INTO link_queue(id, ts, kind, payload, status, edge_json, audit_json, queued_at)"
        " VALUES (?,?,?,?,'queued',NULL,NULL,?)",
        (iid, iso_z(as_of), KIND, json.dumps({"alert_id": alert["id"], "as_of": iso_z(as_of)}), now),
    )
    conn.commit()
    return iid


def get(conn: sqlite3.Connection, iid: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM link_queue WHERE id = ?", (iid,)).fetchone()


def items(conn: sqlite3.Connection, as_of: datetime | None = None) -> list[sqlite3.Row]:
    q, args = "SELECT * FROM link_queue", []
    if as_of is not None:
        q += " WHERE ts <= ?"
        args.append(iso_z(as_of))
    return conn.execute(q + " ORDER BY ts, queued_at, id", args).fetchall()


def depth(conn: sqlite3.Connection, as_of: datetime) -> int:
    """Items still waiting for Earth (queued, edge_done or sent) that exist at as_of."""
    marks = ",".join("?" * len(OPEN_STATUSES))
    return int(conn.execute(
        f"SELECT COUNT(*) FROM link_queue WHERE ts <= ? AND status IN ({marks})", (iso_z(as_of), *OPEN_STATUSES)
    ).fetchone()[0])


def deliverable_items(
    conn: sqlite3.Connection, link: delay_sim.LinkState, now: float, as_of: datetime | None = None
) -> list[sqlite3.Row]:
    """Items that may leave now, in drain order: a drainable status, ts <= as_of (when given), delay rule satisfied."""
    return [
        r for r in items(conn, as_of)
        if r["status"] in DRAINABLE and delay_sim.deliverable(link, r["queued_at"] or 0.0, now)
    ]


def update(conn: sqlite3.Connection, iid: str, **cols) -> None:
    sets = ", ".join(f"{k} = ?" for k in cols)
    conn.execute(f"UPDATE link_queue SET {sets} WHERE id = ?", (*cols.values(), iid))
    conn.commit()


def provenance(a: sqlite3.Row | None) -> dict | None:
    """Truthful badge fields from the edge assessments row: recorded or a live REMOTE API call, and the model that
    ACTUALLY answered. Nothing here is served on the habitat computer, so ON-BOARD never appears: a live run says that
    the remote API stands in for an on-board model. The model name comes from the row, never from a constant."""
    if a is None:
        return None
    recorded = bool(a["recorded"])
    badge = (
        f"RECORDED · EDGE ({a['model']})" if recorded
        else f"EDGE ({a['model']}, remote API: stands in for an on-board model)"
    )
    return {
        "recorded": recorded, "model": a["model"], "provider": a["provider"],
        "edge_location": "recorded" if recorded else "remote_api", "badge": badge,
    }


def view(conn: sqlite3.Connection, r: sqlite3.Row, link: delay_sim.LinkState, now: float) -> dict:
    """The API form of one queue item."""
    payload = json.loads(r["payload"])
    ea = conn.execute("SELECT * FROM assessments WHERE id = ?", (r["edge_assessment_id"],)).fetchone() \
        if r["edge_assessment_id"] else None
    alert = conn.execute("SELECT tray_id FROM alerts WHERE id = ?", (payload["alert_id"],)).fetchone()
    prov = provenance(ea)
    if prov is not None:
        prov["recorded_as_of"] = (payload.get("edge") or {}).get("recorded_as_of")
    secs = delay_sim.seconds_until(link, r["queued_at"] or 0.0, now) if r["status"] in DRAINABLE else None
    return {
        "id": r["id"], "ts": r["ts"], "kind": r["kind"], "alert_id": payload["alert_id"],
        "tray_id": alert["tray_id"] if alert else None, "as_of": payload["as_of"], "status": r["status"],
        "edge_json": json.loads(r["edge_json"]) if r["edge_json"] else None,
        "edge_valid": bool(ea["valid"]) if ea is not None and ea["valid"] is not None else None,
        "edge_assessment_id": r["edge_assessment_id"],
        "provenance": prov,
        "citation_errors": json.loads(ea["citation_errors"] or "[]") if ea is not None else [],
        # null = not evaluated (no edge answer, or the final reply was unparseable); [] = evaluated, nothing ungrounded
        "grounding": json.loads(ea["warnings"]) if ea is not None and ea["warnings"] is not None else None,
        "audit_json": json.loads(r["audit_json"]) if r["audit_json"] else None,
        "earth_assessment_id": r["earth_assessment_id"],
        "deliverable": r["status"] in DRAINABLE and secs == 0.0,
        "deliver_in_s": None if secs is None else round(secs, 1),
        "error": r["error"],
        "perception": PERCEPTION_NOTE,
    }
