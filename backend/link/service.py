"""Link workflows (M8): queue an alert + on-board triage, automatic queueing when the clock advances, and the drain.

enqueue_and_triage  POST /link/queue: queue an open alert, then run the edge triage (free when EDGE_MODE=recorded).
auto_enqueue        the mission clock advanced (prev, new] while the link is not up: queue and triage the alerts whose
                    first_ts falls in that window. Live edge modes triage at most AUTO_MAX_TRIAGE items per advance (the
                    rest stay `queued` without an edge result) so a long clock jump cannot spend the budget.
drain               PAID: for every deliverable item in ts order run the Earth-side Flight Agronomist (about $0.10 each)
                    on the same alert and as_of, then compute the audit in code (backend/link/audit.py).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from backend.agent import agronomist
from backend.api.alerts import visible_alerts
from backend.config import load_settings
from backend.link import audit, delay_sim, edge, queue
from backend.spend import BudgetExceeded
from backend.timeutil import iso_z, parse_ts

log = logging.getLogger(__name__)
AUTO_MAX_TRIAGE = 4
EST_DRAIN_USD = (0.10, 0.20)  # per Earth-side assessment (PROGRESS.md measurements, low effort)
LLMFactory = Callable[[sqlite3.Connection], Any]


def triage_item(conn: sqlite3.Connection, iid: str, llm_factory: LLMFactory | None = None) -> str:
    """Run the edge triage of a queued item and store it. Returns the item's status afterwards.

    No recording, a refused budget or a failed run leave the item `queued` with `error` set (never a fabricated edge)."""
    r = queue.get(conn, iid)
    payload = json.loads(r["payload"])
    as_of = parse_ts(payload["as_of"])
    try:
        aid = edge.triage(conn, payload["alert_id"], as_of, llm_factory=llm_factory)
    except Exception as exc:  # noqa: BLE001  one item must not stop the others
        queue.update(conn, iid, error=str(exc) if isinstance(exc, edge.NoRecording) else f"{type(exc).__name__}: {exc}")
        return "queued"
    edge_json, a = edge.edge_result(conn, aid)
    if a["recorded"]:
        try:
            payload["edge"] = {"recorded_as_of": edge.load_recording(payload["alert_id"], as_of)["as_of"]}
        except edge.NoRecording:
            pass
    errs = json.loads(a["citation_errors"] or "[]")
    if a["valid"] == 1 and edge_json is not None:
        status, error = "edge_done", None
    elif a["valid"] == 0:  # ran but failed validation: keep the answer (if any) and its errors, never edge_done
        status, error = "edge_invalid", "edge triage invalid: " + "; ".join(e[:200] for e in errs[:3])
    else:
        status, error = "queued", (errs[0] if errs else None)
    queue.update(
        conn, iid, edge_assessment_id=aid, edge_json=json.dumps(edge_json) if edge_json is not None else None,
        payload=json.dumps(payload), status=status, error=error,
    )
    return status


def enqueue_and_triage(conn: sqlite3.Connection, alert: dict, as_of: datetime, llm_factory: LLMFactory | None = None,
                       now: float | None = None) -> str:
    iid = queue.enqueue(conn, alert, as_of, time.time() if now is None else now)
    triage_item(conn, iid, llm_factory)
    return iid


def auto_enqueue(conn: sqlite3.Connection, prev: datetime, new: datetime, llm_factory: LLMFactory | None = None,
                 now: float | None = None) -> list[str]:
    """Queue (and triage) the open alerts first raised in (prev, new] if the link is not up. Returns the new item ids."""
    if new <= prev or delay_sim.current(conn).state == "up":
        return []
    firsts = conn.execute(
        "SELECT id FROM alerts WHERE first_ts > ? AND first_ts <= ? ORDER BY first_ts, id", (iso_z(prev), iso_z(new))
    ).fetchall()
    visible = {a["id"]: a for a in visible_alerts(conn, new)}
    live = load_settings().edge_mode != "recorded"
    made: list[str] = []
    for f in firsts:
        a = visible.get(f["id"])
        if a is None or a["status"] != "open":
            continue
        try:
            iid = queue.enqueue(conn, a, new, time.time() if now is None else now)
        except queue.AlreadyQueued:
            continue
        made.append(iid)
        if not live or len(made) <= AUTO_MAX_TRIAGE:
            triage_item(conn, iid, llm_factory)
    return made


def drain(conn: sqlite3.Connection, now: float | None = None, max_items: int | None = None,
          llm_factory: LLMFactory | None = None, as_of: datetime | None = None) -> list[dict]:
    """Deliver the deliverable items in order: Earth-side assessment, then the audit. PAID (one agronomist run per item).

    Returns [{id, status, error}] for the items handled. A refused budget marks that item failed and stops the drain."""
    out: list[dict] = []
    for r in queue.items(conn, as_of):
        if max_items is not None and len(out) >= max_items:
            break
        t = time.time() if now is None else now
        if r["status"] not in queue.DRAINABLE or not delay_sim.deliverable(delay_sim.current(conn), r["queued_at"] or 0.0, t):
            continue
        iid = r["id"]
        payload = json.loads(r["payload"])
        as_of = parse_ts(payload["as_of"])
        queue.update(conn, iid, status="sent", error=None)
        stop = False
        try:
            aid = agronomist.begin(conn, payload["alert_id"], as_of)
            queue.update(conn, iid, earth_assessment_id=aid)
            agronomist.run(conn, aid, as_of, "agronomist", llm_factory=llm_factory)
            a = conn.execute("SELECT json, valid, citation_errors FROM assessments WHERE id = ?", (aid,)).fetchone()
            earth = json.loads(a["json"]) if a["json"] else None
            if earth is None:
                raise RuntimeError("the Earth-side assessment produced no result")
            edge_json = json.loads(r["edge_json"]) if r["edge_json"] else None
            edge_errs: list[str] = []
            if r["edge_assessment_id"]:
                ea = conn.execute(
                    "SELECT valid, citation_errors FROM assessments WHERE id = ?", (r["edge_assessment_id"],)).fetchone()
                if ea is not None and ea["valid"] != 1:
                    edge_errs = json.loads(ea["citation_errors"] or "[]") or ["edge triage failed validation"]
            earth_errs = (json.loads(a["citation_errors"] or "[]") or ["failed validation"]) if a["valid"] != 1 else []
            result = (
                audit.invalid_audit(edge_json, earth, edge_errs, earth_errs) if edge_errs or earth_errs
                else audit.compute_audit(edge_json, earth)
            )
            queue.update(conn, iid, audit_json=json.dumps(result), status="audited")
        except BudgetExceeded as exc:
            queue.update(conn, iid, status="failed", error=f"BudgetExceeded: {exc}")
            stop = True
        except Exception as exc:  # noqa: BLE001
            queue.update(conn, iid, status="failed", error=f"{type(exc).__name__}: {exc}")
        cur = queue.get(conn, iid)
        out.append({"id": iid, "status": cur["status"], "error": cur["error"]})
        if stop:
            break
    return out
