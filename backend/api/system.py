"""GET/POST /clock (mission replay clock) and GET /spend (Meta budget ledger)."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api import deps
from backend.config import load_settings
from backend.sim.clock import CLOCK
from backend.spend import total_spend

router = APIRouter()


class ClockSet(BaseModel):
    as_of: str | None = None
    speed: float | None = Field(None, gt=0, le=1e6, allow_inf_nan=False)
    running: bool | None = None


@router.get("/clock")
def get_clock() -> dict:
    return CLOCK.state()


@router.post("/clock")
def set_clock(body: ClockSet) -> dict:
    try:
        CLOCK.set(body.as_of, body.speed, body.running)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return CLOCK.state()


@router.get("/spend")
def spend(conn: sqlite3.Connection = Depends(deps.get_conn)) -> dict:
    by_role = {
        r["role"]: round(r["usd"], 6)
        for r in conn.execute("SELECT role, SUM(cost_usd) AS usd FROM spend GROUP BY role ORDER BY role")
    }
    return {
        "total_usd": round(total_spend(conn), 6),
        "budget_usd": load_settings().budget_usd,
        "by_role": by_role,
    }
