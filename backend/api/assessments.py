"""POST /alerts/{id}/assess, GET /assessments/{id} and the SSE stream (M4).

POST creates the assessments row, starts the agent in a background thread (own DB connection) and returns the
id at once. The stream replays the events logged so far (tool_log rows, in order) and then follows the run by
polling the DB until the assessment finishes (an `assessment` event) or fails (an `error` event); a stream
opened after the run replays the same sequence. Assessments are hidden by `as_of` like every other read.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from backend.agent import agronomist
from backend.agent.llm_client import LLMClient
from backend.api import deps
from backend.timeutil import iso_z

router = APIRouter()
log = logging.getLogger(__name__)

POLL_S = 0.15
STREAM_TIMEOUT_S = 600
STALL_S = 120  # a running assessment with no heartbeat for this long is reported as stalled
START_LOCK = threading.Lock()
THREADS: dict[str, threading.Thread] = {}  # assessment_id -> worker (tests join it)


def llm_factory(conn: sqlite3.Connection) -> LLMClient:
    """Seam for tests: they replace this with a function returning a scripted fake client."""
    return LLMClient(conn)


def _worker(aid: str, as_of: datetime, role: str) -> None:
    conn = deps.open_db()
    try:
        agronomist.run(conn, aid, as_of, role, llm_factory=llm_factory)
    except Exception:  # noqa: BLE001  already recorded on the assessment row and in the error event
        log.warning("assessment %s ended with an error", aid)
    finally:
        conn.close()


@router.post("/alerts/{alert_id}/assess")
def start_assessment(
    alert_id: str,
    force: bool = False,
    as_of: datetime = Depends(deps.get_as_of),
    conn: sqlite3.Connection = Depends(deps.get_conn),
) -> dict:
    """Start a run. For the same alert and as_of, an in-flight or finished run is returned instead (?force=1 starts anew)."""
    with START_LOCK:  # check + insert atomically so a double click cannot start two paid runs
        if not force:
            existing = conn.execute(
                "SELECT id FROM assessments WHERE alert_id = ? AND ts = ? AND (valid IS NULL OR json IS NOT NULL"
                " OR citation_errors NOT LIKE '[\"error:%') ORDER BY rowid DESC LIMIT 1",
                (alert_id, iso_z(as_of)),
            ).fetchone()
            if existing:
                return {"assessment_id": existing["id"]}
        try:
            aid = agronomist.begin(conn, alert_id, as_of)
        except agronomist.AlertNotFound as e:
            raise HTTPException(404, str(e)) from None
    t = threading.Thread(target=_worker, args=(aid, as_of, "agronomist"), daemon=True, name=f"assess-{aid}")
    THREADS[aid] = t
    t.start()
    return {"assessment_id": aid}


def _row(conn: sqlite3.Connection, aid: str) -> sqlite3.Row:
    r = conn.execute("SELECT * FROM assessments WHERE id = ?", (aid,)).fetchone()
    if r is None:
        raise HTTPException(404, f"unknown assessment {aid!r}")
    return r


@router.get("/assessments/{assessment_id}")
def get_assessment(
    assessment_id: str,
    as_of: datetime = Depends(deps.get_as_of),
    conn: sqlite3.Connection = Depends(deps.get_conn),
) -> dict:
    r = _row(conn, assessment_id)
    if r["ts"] > iso_z(as_of):
        raise HTTPException(404, f"assessment {assessment_id!r} is later than as_of")
    actions = [
        {"action_id": a["id"], "status": a["status"], "proposal": json.loads(a["proposal_json"] or "{}")}
        for a in conn.execute("SELECT * FROM actions WHERE assessment_id = ? ORDER BY ts, id", (assessment_id,))
    ]
    return {
        "assessment_id": r["id"],
        "tray_id": r["tray_id"],
        "alert_id": r["alert_id"],
        "assessment": json.loads(r["json"]) if r["json"] else None,
        "valid": bool(r["valid"]) if r["valid"] is not None else None,
        "citation_errors": json.loads(r["citation_errors"] or "[]"),
        "latency_ms": r["latency_ms"],
        "cost_usd": r["cost_usd"],
        "recorded": bool(r["recorded"]),
        "model": r["model"],
        "provider": r["provider"],
        "actions": actions,
    }


def _tool_events(rows: list[sqlite3.Row]) -> list[dict]:
    out = []
    for t in rows:
        out.append({"event": "tool_call", "data": json.dumps(
            {"round": t["round"], "tool": t["tool"], "args": json.loads(t["args"] or "{}")})})
        out.append({"event": "tool_result", "data": json.dumps(
            {"round": t["round"], "tool": t["tool"], "ids": json.loads(t["returned_ids"] or "[]")})})
    return out


@router.get("/assessments/{assessment_id}/stream")
async def stream(
    assessment_id: str,
    as_of: datetime = Depends(deps.get_as_of),
    conn: sqlite3.Connection = Depends(deps.get_conn),
) -> EventSourceResponse:
    if _row(conn, assessment_id)["ts"] > iso_z(as_of):  # 404 before the stream starts
        raise HTTPException(404, f"assessment {assessment_id!r} is later than as_of")

    async def events():
        c = deps.open_db()
        sent = 0
        deadline = time.monotonic() + STREAM_TIMEOUT_S
        try:
            while True:
                a = c.execute("SELECT * FROM assessments WHERE id = ?", (assessment_id,)).fetchone()
                done = a is not None and a["valid"] is not None  # read BEFORE tool_log: its rows are complete then
                rows = c.execute(
                    "SELECT round, tool, args, returned_ids FROM tool_log WHERE assessment_id = ? ORDER BY rowid",
                    (assessment_id,),
                ).fetchall()
                for ev in _tool_events(rows[sent:]):
                    yield ev
                sent = len(rows)
                if done:
                    errs = json.loads(a["citation_errors"] or "[]")
                    if a["json"] is None and errs and errs[0].startswith("error: "):
                        yield {"event": "error", "data": json.dumps({"message": errs[0][len("error: "):]})}
                    else:
                        yield {"event": "assessment", "data": json.dumps({
                            "assessment_id": assessment_id, "valid": bool(a["valid"]),
                            "latency_ms": a["latency_ms"], "recorded": bool(a["recorded"])})}
                    return
                if a is not None and time.time() - float(a["updated_at"] or 0) > STALL_S:
                    yield {"event": "error", "data": json.dumps({"message": "run stalled"})}
                    return
                if time.monotonic() > deadline:
                    yield {"event": "error", "data": json.dumps({"message": "stream timed out"})}
                    return
                await asyncio.sleep(POLL_S)
        finally:
            c.close()

    return EventSourceResponse(events())


