"""POST /actions/{id}: a human approves or rejects an agent-proposed action."""

from __future__ import annotations

import sqlite3
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api import deps

router = APIRouter()
STATUS = {"approve": "approved", "reject": "rejected"}


class Decision(BaseModel):
    decision: Literal["approve", "reject"]


@router.post("/actions/{action_id}")
def decide(action_id: str, body: Decision, conn: sqlite3.Connection = Depends(deps.get_conn)) -> dict:
    status = STATUS[body.decision]
    cur = conn.execute("UPDATE actions SET status = ? WHERE id = ?", (status, action_id))
    conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, f"unknown action {action_id!r}")
    return {"action_id": action_id, "status": status}
