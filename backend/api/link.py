"""Mars mode: GET/POST /link, GET/POST /link/queue, POST /link/drain (M8).

The link is SIMULATED (`simulated: true`); the real one-way light delay is text only. Reads are time-gated by `as_of`
(the link state is the last change whose mission time is <= as_of; queue items with ts <= as_of). Queueing an alert runs
the on-board edge triage (free when EDGE_MODE=recorded). POST /link/drain is the PAID step (one Flight Agronomist run per
item, about $0.10): it needs `{"confirm": true}`, is refused past the budget, and runs in a background thread; poll
GET /link/queue for the statuses. Without `confirm` it only returns the plan and the estimated cost.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.agent import replay as agent_replay
from backend.agent.llm_client import resolve_role
from backend.api import assessments as assess_api
from backend.api import deps
from backend.api.alerts import visible_alerts
from backend.config import load_settings
from backend.link import delay_sim, edge, queue, service
from backend.sim.clock import CLOCK
from backend.spend import BudgetExceeded, check_budget

router = APIRouter()
LOCK = threading.Lock()  # one enqueue / drain start at a time
DRAIN_THREADS: list[threading.Thread] = []  # tests join these


def _llm(conn: sqlite3.Connection):
    """Resolved at call time so tests can swap assessments.llm_factory."""
    return assess_api.llm_factory(conn)


def link_view(conn: sqlite3.Connection, as_of: datetime) -> dict:
    st = delay_sim.state_at(conn, as_of)
    settings = load_settings()
    return {
        "state": st.state, "delay_s": st.delay_s, "since": st.since, "queue_depth": queue.depth(conn, as_of),
        "simulated": True, "light_delay_note": delay_sim.LIGHT_DELAY_NOTE,
        "edge_mode": settings.edge_mode, "edge_model": edge.edge_model_name(),
        # recorded = a stored model response is replayed; remote_api = a live API call. Neither is an on-board model.
        "edge_location": "recorded" if settings.edge_mode == "recorded" else "remote_api",
        "earth_model": resolve_role("agronomist").model,
        "edge_recordings": edge.recorded_alert_ids() if settings.edge_mode == "recorded" else [],
        # DEMO_MODE: every model result is replayed; these alerts have a recorded Earth-side run (others answer 409)
        "demo_mode": settings.demo_mode,
        "agent_recordings": agent_replay.recorded_alert_ids() if settings.demo_mode else [],
    }


def run_auto(conn: sqlite3.Connection, prev: datetime, new: datetime) -> list[str]:
    return service.auto_enqueue(conn, prev, new, llm_factory=_llm)


class LinkSet(BaseModel):
    state: Literal["up", "delayed", "down"]
    delay_s: float | None = Field(None, allow_inf_nan=False)


class EnqueueBody(BaseModel):
    alert_id: str


class DrainBody(BaseModel):
    confirm: bool = False
    max_items: int | None = Field(None, ge=1, le=20)


@router.get("/link")
def get_link(as_of: datetime = Depends(deps.get_as_of), conn: sqlite3.Connection = Depends(deps.get_conn)) -> dict:
    return link_view(conn, as_of)


@router.post("/link")
def set_link(body: LinkSet, conn: sqlite3.Connection = Depends(deps.get_conn)) -> dict:
    now = CLOCK.now()
    try:
        delay_sim.transition(conn, body.state, body.delay_s, now)
    except delay_sim.InvalidTransition as e:
        raise HTTPException(409, str(e)) from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return link_view(conn, now)


@router.get("/link/queue")
def get_queue(as_of: datetime = Depends(deps.get_as_of), conn: sqlite3.Connection = Depends(deps.get_conn)) -> dict:
    link, now = delay_sim.current(conn), time.time()
    return {"items": [queue.view(conn, r, link, now) for r in queue.items(conn, as_of)]}


@router.post("/link/queue")
def post_queue(
    body: EnqueueBody, as_of: datetime = Depends(deps.get_as_of), conn: sqlite3.Connection = Depends(deps.get_conn)
) -> dict:
    """Queue an alert that is open at as_of and run its on-board triage. 404 unknown/not yet raised, 409 resolved,
    already queued, or the link is up (use POST /alerts/{id}/assess)."""
    alert = next((a for a in visible_alerts(conn, as_of) if a["id"] == body.alert_id), None)
    if alert is None:
        raise HTTPException(404, f"alert {body.alert_id!r} is unknown or not yet raised at as_of")
    if alert["status"] != "open":
        raise HTTPException(409, f"alert {body.alert_id!r} is not open at as_of")
    if delay_sim.current(conn).state == "up":
        raise HTTPException(409, "the link is up: nothing to queue (POST /alerts/{id}/assess reaches Earth directly)")
    with LOCK:
        try:
            iid = service.enqueue_and_triage(conn, alert, as_of, llm_factory=_llm)
        except queue.AlreadyQueued as e:
            raise HTTPException(409, str(e)) from None
    return queue.view(conn, queue.get(conn, iid), delay_sim.current(conn), time.time())


def _drain_worker(max_items: int | None, as_of: datetime) -> None:
    conn = deps.open_db()
    try:
        service.drain(conn, max_items=max_items, llm_factory=_llm, as_of=as_of)
    finally:
        conn.close()


@router.post("/link/drain")
def post_drain(
    body: DrainBody | None = None, as_of: datetime = Depends(deps.get_as_of),
    conn: sqlite3.Connection = Depends(deps.get_conn),
) -> dict:
    """Plan (no `confirm`) or start (`confirm: true`) the paid drain of the deliverable items. Like the reads, only
    items with ts <= as_of are listed and drained."""
    body = body or DrainBody()
    link = delay_sim.current(conn)
    if link.state == "down":
        raise HTTPException(409, "the link is down: nothing can be delivered")
    due = queue.deliverable_items(conn, link, time.time(), as_of)
    if body.max_items:
        due = due[: body.max_items]
    lo, hi = (0.0, 0.0) if load_settings().demo_mode else service.EST_DRAIN_USD  # DEMO_MODE replays: free
    out = {
        "started": False, "confirm_required": not body.confirm, "link_state": link.state,
        "item_ids": [r["id"] for r in due], "est_cost_usd": {"low": round(len(due) * lo, 2), "high": round(len(due) * hi, 2)},
        "detail": "",
    }
    if not due:
        out["detail"] = "nothing is deliverable yet" + (" (delay not elapsed)" if link.state == "delayed" else "")
        return out
    if not body.confirm:
        out["detail"] = "send {\"confirm\": true} to run the Earth-side agronomist on these items (paid)"
        return out
    with LOCK:
        if any(t.is_alive() for t in DRAIN_THREADS):
            raise HTTPException(409, "a drain is already running")
        try:
            if hi:
                check_budget(conn, len(due) * hi)
        except BudgetExceeded as e:
            raise HTTPException(402, str(e)) from None
        t = threading.Thread(target=_drain_worker, args=(body.max_items, as_of), daemon=True, name="link-drain")
        DRAIN_THREADS[:] = [t]
        t.start()
    out.update(started=True, detail="drain started; poll GET /link/queue for item statuses")
    return out
