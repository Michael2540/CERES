"""Flight Agronomist orchestrator (M4b).

assess() = begin() (creates the assessments row) + run():
  round 0   pre-fetch get_tray_overview for the alerted tray and its reference, through the SAME logged handlers;
  rounds 1-5  tool loop, tool_choice auto; all tool messages of a round come first, then ONE user message with
            the images for the frame_refs returned that round (images are only legal in user messages). The image
            cap is spread round-robin over the trays of the round; a frame whose image was not attached is marked
            "attached": false in its tool message and is not citable;
  final     if the loop ended with a plain assistant answer that parses and validates, it is used as is; otherwise
            a structured call (tools omitted, strict json_schema) and, on validator errors, ONE retry.
Events (tool_call, tool_result, assessment, error) go to `on_event(event, data)` and are replayable from tool_log
and the assessments row (the SSE endpoint polls those). A failed or invalid run supersedes its pending actions.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import sqlite3
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from backend.agent import grounding
from backend.agent.llm_client import ChatResult, LLMClient, resolve_role
from backend.agent.tools import TOOL_SCHEMAS, ToolContext, log_error_call, run_tool, set_logged_ids
from backend.agent.validate import assessment_schema, validate
from backend.api.alerts import visible_alerts
from backend.config import ROOT, load_trays
from backend.timeutil import iso_z

log = logging.getLogger(__name__)

PROMPTS = ROOT / "prompts"
MAX_ROUNDS = 5
MAX_CALLS_PER_ROUND = 8
MAX_IMAGE_WIDTH = 640
Event = Callable[[str, dict], None]

NUDGE = (
    "Investigation complete. Tools are no longer available. Return the final assessment as JSON matching the "
    "schema. Cite only ids that tools returned (or the alert id). Every action you recommend must already have "
    "been proposed with propose_action; otherwise leave recommended_actions empty."
)


class AlertNotFound(LookupError):
    pass


def _render(template: str, values: dict[str, str]) -> str:
    return re.sub(r"\{\{(\w+)\}\}", lambda m: values.get(m.group(1), m.group(0)), template)


def response_format() -> dict:
    return {"type": "json_schema", "json_schema": {"name": "assessment", "strict": True, "schema": assessment_schema()}}


def find_alert(conn: sqlite3.Connection, alert_id: str, as_of: datetime) -> dict:
    a = next((x for x in visible_alerts(conn, as_of) if x["id"] == alert_id), None)
    if a is None:
        raise AlertNotFound(f"alert {alert_id!r} is unknown or not yet raised at {iso_z(as_of)}")
    return a


def touch(conn: sqlite3.Connection, aid: str) -> None:
    """Heartbeat for the stall check of the SSE stream."""
    conn.execute("UPDATE assessments SET updated_at = ? WHERE id = ?", (time.time(), aid))
    conn.commit()


def image_part(path: str | Path) -> dict | None:
    """JPEG data-URL part, downscaled to <= MAX_IMAGE_WIDTH px wide. None if the file cannot be read."""
    img = cv2.imread(str(ROOT / path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > MAX_IMAGE_WIDTH:
        img = cv2.resize(img, (MAX_IMAGE_WIDTH, round(h * MAX_IMAGE_WIDTH / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf).decode()}}


def allocate_images(
    conn: sqlite3.Connection, refs: list[dict], used: set[str], cap: int
) -> tuple[list[dict], list[str], set[str]]:
    """Spend the remaining image cap round-robin over the trays of `refs`.

    Returns (image parts, labels, ids whose image the model has now seen). Frames attached in an earlier round count
    as seen without using the cap. `used` is updated with the newly attached ids."""
    queues: dict[str, deque] = {}
    for r in refs:
        queues.setdefault(r["tray_id"], deque()).append(r)
    parts: list[dict] = []
    labels: list[str] = []
    seen: set[str] = {r["id"] for r in refs if r["id"] in used}  # attached in an earlier round
    progress = True
    while progress:
        progress = False
        for q in queues.values():  # one new image per tray per pass
            while q:
                ref = q.popleft()
                if ref["id"] in used:
                    continue
                if len(used) >= cap:
                    q.clear()
                    break
                row = conn.execute("SELECT path FROM frames WHERE id = ?", (ref["id"],)).fetchone()
                part = image_part(row["path"]) if row else None
                if part is None:
                    continue  # unreadable file: try the next frame of this tray
                used.add(ref["id"])
                seen.add(ref["id"])
                parts.append(part)
                labels.append(f"{ref['id']} (tray {ref['tray_id']}, {ref['ts']})")
                progress = True
                break
    return parts, labels, seen


def begin(conn: sqlite3.Connection, alert_id: str, as_of: datetime, role: str = "agronomist") -> str:
    """Create the assessments row (model, provider, recorded=false) and return its id."""
    alert = find_alert(conn, alert_id, as_of)
    cfg = resolve_role(role)
    base = f"asmt_{alert['tray_id']}_{iso_z(as_of).replace('-', '').replace(':', '')}"
    aid, k = base, 1
    while conn.execute("SELECT 1 FROM assessments WHERE id = ?", (aid,)).fetchone():
        k += 1
        aid = f"{base}_{k}"
    conn.execute(
        "INSERT INTO assessments(id, ts, tray_id, alert_id, model, provider, json, citation_errors, valid,"
        " latency_ms, cost_usd, recorded, updated_at, role) VALUES (?,?,?,?,?,?,NULL,NULL,NULL,NULL,0,0,?,?)",
        (aid, iso_z(as_of), alert["tray_id"], alert_id, cfg.model, cfg.provider, time.time(), role),
    )
    conn.commit()
    return aid


def _supersede(conn: sqlite3.Connection, aid: str) -> None:
    conn.execute("UPDATE actions SET status = 'superseded' WHERE assessment_id = ? AND status = 'pending'", (aid,))


def _fail(conn: sqlite3.Connection, aid: str, msg: str, t0: float, cost: float) -> None:
    conn.execute(
        "UPDATE assessments SET json = NULL, citation_errors = ?, valid = 0, latency_ms = ?, cost_usd = ? WHERE id = ?",
        (json.dumps([f"error: {msg}"]), int((time.perf_counter() - t0) * 1000), cost, aid),
    )
    _supersede(conn, aid)
    conn.commit()


def assess(
    conn: sqlite3.Connection,
    alert_id: str,
    as_of: datetime | None = None,
    role: str = "agronomist",
    on_event: Event | None = None,
    llm: Any | None = None,
) -> str:
    """Investigate one alert and persist the assessment. Returns the assessment id."""
    if as_of is None:
        from backend.sim.clock import CLOCK

        as_of = CLOCK.now()
    aid = begin(conn, alert_id, as_of, role)
    run(conn, aid, as_of, role, on_event, llm)
    return aid


def run(
    conn: sqlite3.Connection,
    aid: str,
    as_of: datetime,
    role: str = "agronomist",
    on_event: Event | None = None,
    llm: Any | None = None,
    llm_factory: Callable[[sqlite3.Connection], Any] | None = None,
) -> None:
    emit = on_event or (lambda _e, _d: None)
    t0 = time.perf_counter()
    try:
        llm = llm or (llm_factory or LLMClient)(conn)  # inside the try: a missing key is recorded, not lost
        row = conn.execute("SELECT alert_id FROM assessments WHERE id = ?", (aid,)).fetchone()
        alert = find_alert(conn, row["alert_id"], as_of)
        _investigate(conn, aid, alert, as_of, role, emit, llm, t0)
    except Exception as exc:
        log.exception("assessment %s failed", aid)
        _fail(conn, aid, f"{type(exc).__name__}: {exc}", t0, _total_cost(conn, aid))
        emit("error", {"message": f"{type(exc).__name__}: {exc}"})
        raise


def _total_cost(conn: sqlite3.Connection, aid: str) -> float:
    r = conn.execute("SELECT cost_usd FROM assessments WHERE id = ?", (aid,)).fetchone()
    return float(r["cost_usd"] or 0.0) if r else 0.0


class _Meter:
    """Sums the cost of each chat call into assessments.cost_usd as it happens (so a failure keeps the total)."""

    def __init__(self, conn: sqlite3.Connection, aid: str, llm: Any, role: str) -> None:
        self.conn, self.aid, self.llm, self.role = conn, aid, llm, role
        self.cost = 0.0

    def chat(self, messages: list[dict], tools=None, response_format=None) -> ChatResult:
        touch(self.conn, self.aid)
        res = self.llm.chat(self.role, messages, tools=tools, response_format=response_format)
        self.cost += res.cost_usd
        self.conn.execute(
            "UPDATE assessments SET cost_usd = ?, updated_at = ? WHERE id = ?", (self.cost, time.time(), self.aid)
        )
        self.conn.commit()
        return res


def _reject_constant(name: str):
    raise ValueError(f"non-finite number {name} is not allowed")


def _parse_json(text: str) -> tuple[dict | None, str | None]:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        obj = json.loads(t, parse_constant=_reject_constant)
    except json.JSONDecodeError as e:
        return None, f"final answer is not valid JSON ({e.msg} at char {e.pos})"
    except ValueError as e:
        return None, f"final answer is not valid JSON ({e})"
    return (obj, None) if isinstance(obj, dict) else (None, "final answer must be a JSON object")


def _investigate(
    conn: sqlite3.Connection, aid: str, alert: dict, as_of: datetime, role: str, emit: Event, llm: Any, t0: float
) -> None:
    cfg = resolve_role(role)
    trays = {t.id: t for t in load_trays().trays}
    tray_id = alert["tray_id"]
    focus = trays["A"] if tray_id == "air" else trays[tray_id]  # a shared air-sensor fault uses tray A as context
    ref = focus.reference_tray_id
    ctx = ToolContext(conn, aid, as_of, 0, alert["id"], alert["rule"])
    meter = _Meter(conn, aid, llm, role)

    def call(tool: str, args: dict, rnd: int, announce: bool = True) -> tuple[dict, list[str]]:
        ctx.round = rnd
        if announce:
            emit("tool_call", {"round": rnd, "tool": tool, "args": args})
        result, ids = run_tool(ctx, tool, args)
        touch(conn, aid)
        if announce:
            emit("tool_result", {"round": rnd, "tool": tool, "ids": ids})
        return result, ids

    over_focus, _ = call("get_tray_overview", {"tray_id": focus.id}, 0)
    over_ref = call("get_tray_overview", {"tray_id": ref}, 0)[0] if ref else over_focus
    reference_desc = f"tray {ref}, same crop, control" if ref else f"tray {focus.id}'s own earlier baseline"
    user = _render(
        (PROMPTS / "agronomist.user.md").read_text(encoding="utf-8"),
        {
            "alert_id": alert["id"], "rule": alert["rule"], "tray_id": tray_id,
            "crop": "Arabidopsis thaliana, 20-plant phenotyping tray", "treatment": over_focus["treatment"],
            "alert_ts": alert["ts"], "threshold_tag": alert["threshold_tag"] or "",
            "alert_evidence_ids": ", ".join(alert["evidence_ids"]), "reference_desc": reference_desc,
            "as_of": iso_z(as_of),
            "overview_alerted_json": json.dumps(over_focus, separators=(",", ":")),
            "overview_reference_json": json.dumps(over_ref, separators=(",", ":")) if ref else "",
        },
    )
    messages: list[dict] = [
        {"role": "system", "content": (PROMPTS / "agronomist.system.md").read_text(encoding="utf-8")},
        {"role": "user", "content": user},
    ]
    used_frames: set[str] = set()
    alert_row = {"id": alert["id"], "rule": alert["rule"], "tray_id": tray_id, "ts": alert["ts"]}
    assessment: dict | None = None
    errors: list[str] = []
    done = False

    for rnd in range(1, MAX_ROUNDS + 1):
        res = meter.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(res.message)
        calls = res.message.get("tool_calls") or []
        if not calls:
            direct, perr = _parse_json(res.message.get("content") or "")
            if direct is not None and not perr and not validate(direct, aid, alert_row, conn, as_of):
                assessment, errors, done = direct, [], True  # a valid plain answer: skip the structured call
            break
        ctx.round = rnd
        run_calls: list[tuple[dict, str, dict, dict, list[str], int | None]] = []
        for i, tc in enumerate(calls):
            fn = tc.get("function") or {}
            name = fn.get("name") or "?"
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise TypeError("arguments must be a JSON object")
                err = f"at most {MAX_CALLS_PER_ROUND} tool calls per round" if i >= MAX_CALLS_PER_ROUND else None
            except (ValueError, TypeError) as e:
                args, err = {}, f"bad tool arguments: {e}"
            if err:
                log_error_call(ctx, name, err)
                run_calls.append((tc, name, args, {"error": err}, [], None))
            else:
                result, ids = call(name, args, rnd, announce=False)
                run_calls.append((tc, name, args, result, ids, ctx.last_log_rowid))
        refs = {r["id"]: r for _tc, _n, _a, res_, _i, _r in run_calls for r in res_.get("frame_refs", [])}
        parts, labels, seen = allocate_images(conn, list(refs.values()), used_frames, cfg.image_cap)
        for _tc, name, args, result, ids, rowid in run_calls:
            if "frame_refs" in result:
                for r in result["frame_refs"]:
                    r["attached"] = r["id"] in seen
                miss = [r["id"] for r in result["frame_refs"] if not r["attached"]]
                result["note"] = (
                    "Images for the attached frames follow in the next user message."
                    + (f" NOT attached (image limit reached or unreadable), do not describe or cite: {miss}." if miss else "")
                )
                ids = [i for i in ids if i not in miss]
                set_logged_ids(ctx, rowid, ids)
            emit("tool_call", {"round": rnd, "tool": name, "args": args})
            emit("tool_result", {"round": rnd, "tool": name, "ids": ids})
        for tc, _n, _a, result, _i, _r in run_calls:
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": json.dumps(result, separators=(",", ":"))})
        if parts:  # AFTER all tool messages of the round
            messages.append({"role": "user", "content": [
                {"type": "text", "text": f"Images for the frames attached in round {rnd}, in order: " + "; ".join(labels) + "."},
                *parts]})

    if not done:
        messages.append({"role": "user", "content": NUDGE})
        fmt = response_format()
        res = meter.chat(messages, tools=None, response_format=fmt)
        assessment, perr = _parse_json(res.message.get("content") or "")
        errors = [perr] if perr else validate(assessment, aid, alert_row, conn, as_of)
        if errors:  # retry the final call ONCE with the errors fed back
            messages.append(res.message)
            messages.append({
                "role": "user",
                "content": "Your answer failed validation:\n- " + "\n- ".join(errors)
                + "\nReturn the corrected full JSON. Tools are not available: cite only ids tools already returned (or the "
                "alert id), keep likelihoods summing to at most 1, and drop any action that was not proposed.",
            })
            res = meter.chat(messages, tools=None, response_format=fmt)
            retry, perr = _parse_json(res.message.get("content") or "")
            if perr:
                errors = [perr]
            else:
                assessment, errors = retry, validate(retry, aid, alert_row, conn, as_of)

    latency = int((time.perf_counter() - t0) * 1000)
    conn.execute(
        "UPDATE assessments SET json = ?, citation_errors = ?, valid = ?, latency_ms = ?, cost_usd = ?, recorded = 0"
        " WHERE id = ?",
        (json.dumps(assessment) if assessment is not None else None, json.dumps(errors), 0 if errors else 1,
         latency, meter.cost, aid),
    )
    if errors:
        _supersede(conn, aid)
    conn.commit()
    try:  # numeric grounding is a WARNING on the row; it never changes validity or the retry logic
        grounding.store_warnings(conn, aid, assessment, alert)
    except Exception:
        log.exception("grounding check failed for %s", aid)
    emit("assessment", {"assessment_id": aid, "valid": not errors, "latency_ms": latency, "recorded": False})
