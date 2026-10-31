"""FastAPI entry point: uvicorn backend.app:app

A background task advances the pipeline while the mission clock runs: at each new mission hour it calls
pipeline.run_cycle (no segmentation unless CERES_LIVE_SEGMENT=1, which uses the api SAM backend; the budget
guard still applies). CERES_DISABLE_BG=1 turns the task off (tests).
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api import (
    actions,
    alerts,
    assessments,
    deps,
    evaluation,
    evidence,
    frames,
    link,
    series,
    system,
    trays,
)
from backend.config import load_settings, load_trays
from backend.perception.sam_client import get_backend
from backend.pipeline import run_cycle
from backend.sim.clock import CLOCK, MissionClock

log = logging.getLogger("ceres.bg")
MAX_CATCHUP_H = 24 * 40  # after a jump forward, replay at most this many hourly cycles (~0.13 s each)


def _hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def cycle_once(as_of: datetime) -> None:
    """One pipeline cycle on its own connection (runs in a worker thread)."""
    conn = deps.open_db()
    try:
        live_seg = os.getenv("CERES_LIVE_SEGMENT") == "1"
        backend = get_backend("api", conn) if live_seg else None
        run_cycle(conn, load_trays(), as_of, backend, segment=live_seg)
    finally:
        conn.close()


def link_advance(prev: datetime, new: datetime) -> None:
    """The clock advanced (prev, new]: queue and triage the alerts raised in it while the link is not up (own connection)."""
    from backend.api import link as link_api

    conn = deps.open_db()
    try:
        link_api.run_auto(conn, prev, new)
    finally:
        conn.close()


async def background_loop(clock: MissionClock, poll_s: float = 1.0) -> None:
    last: datetime | None = None
    while True:
        try:
            if clock.running:
                cur = _hour(clock.now())
                if last is None or cur < last:
                    last = cur - timedelta(hours=1)  # first run, or the clock went back
                t = max(last + timedelta(hours=1), cur - timedelta(hours=MAX_CATCHUP_H))
                while t <= cur:
                    try:
                        await asyncio.to_thread(cycle_once, t)  # in a worker thread: requests are not blocked
                    except asyncio.CancelledError:
                        raise
                    except Exception:  # one bad hour must not stall the loop
                        log.exception("background cycle failed at %s", t)
                    last = t  # advance even after a failure
                    t += timedelta(hours=1)
                last = max(last, cur)
        except asyncio.CancelledError:
            raise
        except Exception:  # a failed cycle must not kill the loop
            log.exception("background cycle failed")
        try:  # Mars mode: alerts first raised in the span the clock just crossed (also after a paused jump)
            adv = clock.take_advance()
            if adv:
                await asyncio.to_thread(link_advance, *adv)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("link auto-queue failed")
        await asyncio.sleep(poll_s)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    deps.open_db().close()  # make sure the schema exists
    task = None
    if os.getenv("CERES_DISABLE_BG") != "1":
        task = asyncio.create_task(background_loop(CLOCK))
    try:
        yield
    finally:
        if task:
            task.cancel()


app = FastAPI(title="CERES", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3100"],
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.exception_handler(RequestValidationError)
async def _validation_error(_req: Request, exc: RequestValidationError) -> JSONResponse:
    """422 without echoing the offending input (inf/NaN are not JSON serializable)."""
    errs = [{k: v for k, v in e.items() if k not in ("input", "ctx")} for e in exc.errors()]
    return JSONResponse({"detail": errs}, status_code=422)


for module in (frames, trays, series, alerts, evidence, actions, system, assessments, evaluation, link):
    app.include_router(module.router)


@app.get("/health")
def health() -> dict:
    s = load_settings()
    return {"status": "ok", "demo_mode": s.demo_mode, "edge_mode": s.edge_mode}
