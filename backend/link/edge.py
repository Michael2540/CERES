"""On-board edge triage (M8): the Edge Sentinel that works while the Earth link is down or delayed.

Same orchestrator style as the Flight Agronomist (backend/agent/agronomist.py): every tool call goes through the logged
handlers (`tool_log` rows with args, returned ids and results), `as_of` is injected and never chosen by the model, and the
run writes an `assessments` row with role=edge, the provider and the model that ACTUALLY answered.

Tools: get_tray_overview, get_sensor_series, compute_vpd, plus at most ONE tray image (the latest frame at or before
as_of, fetched by the orchestrator through get_frames so its id is citable only if the image was attached).

PERCEPTION IS NEVER RE-RUN in Mars mode: the edge works on frames already segmented before the link dropped (the queue
API says so in `perception`). There is no SAM call anywhere in this module.

Flow: round 0 prefetch (overview, frame) -> up to MAX_ROUNDS tool rounds -> structured final answer (strict edge.json
schema). The answer is checked by validate.validate_edge (schema, citations, tray) and grounding (numbers in the two free
text fields must come from this run's tool results). On any error ONE repair call gets the errors; after that the result is
stored with valid=false. The whole prompt is kept under MAX_PROMPT_TOKENS (llm_client.estimate_tokens): an oversize start
is refused (TokenBudgetExceeded), and a tool result that would not fit is replaced by an "omitted" stub and its ids are
removed from the citable set.

EDGE_MODE: spark_contributor and remote_glimmer are live calls through LLMClient (budget guard, ledger). `recorded`
replays data/demo/edge_runs/<alert_id>.json: the file holds the model responses AND the tool results, the replay runs the
same orchestrator and validators against them, writes the same assessments and tool_log rows, and never builds an LLM
client. With no usable recording it raises NoRecording; nothing falls back to a live or invented result.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.agent import grounding
from backend.agent.agronomist import PROMPTS, _Meter, _parse_json, _render, find_alert, image_part
from backend.agent.llm_client import ChatResult, LLMClient, estimate_tokens, resolve_role
from backend.agent.tools import (
    TOOL_SCHEMAS,
    ToolContext,
    log_error_call,
    log_tool_call,
    run_tool,
    set_logged_ids,
)
from backend.agent.validate import edge_schema, validate_edge
from backend.config import ROOT, load_settings, load_trays
from backend.link import delay_sim
from backend.timeutil import iso_z, parse_ts

log = logging.getLogger(__name__)

RUNS_DIR = ROOT / "data" / "demo" / "edge_runs"
RECORDING_VERSION = 1
EDGE_TOOLS = ("get_tray_overview", "get_sensor_series", "compute_vpd")
EDGE_TOOL_SCHEMAS = [t for t in TOOL_SCHEMAS if t["function"]["name"] in EDGE_TOOLS]
MAX_PROMPT_TOKENS = 16000
FINAL_RESERVE_TOKENS = 1500  # kept free so the final (and repair) call always fits
MAX_ROUNDS = 3
MAX_CALLS_PER_ROUND = 6
NO_RECORDING = "no recorded edge run for this alert"

NUDGE = (
    "Triage complete. Tools are no longer available. Return the triage as JSON matching the schema. Cite only ids "
    "that tools returned (or the alert id). Use only numbers a tool returned; do not compute or round numbers yourself."
)


class NoRecording(LookupError):
    """EDGE_MODE=recorded and there is no (usable) recorded run for the alert."""


class TokenBudgetExceeded(RuntimeError):
    """The edge prompt would exceed MAX_PROMPT_TOKENS; the call is refused."""


class ReplayMismatch(RuntimeError):
    """The recording does not fit the code path that replays it (changed tools or prompts): not replayed."""


def response_format() -> dict:
    return {"type": "json_schema", "json_schema": {"name": "edge_triage", "strict": True, "schema": edge_schema()}}


# ---------------------------------------------------------------- recordings
def recording_path(alert_id: str, runs_dir: Path | None = None) -> Path:
    return Path(runs_dir or RUNS_DIR) / f"{alert_id}.json"


def recorded_models(runs_dir: Path | None = None) -> list[str]:
    """Distinct model names inside the recordings on disk (for GET /link)."""
    out: set[str] = set()
    for p in Path(runs_dir or RUNS_DIR).glob("*.json"):
        try:
            out.add(json.loads(p.read_text(encoding="utf-8"))["model"])
        except (OSError, ValueError, KeyError):
            continue
    return sorted(out)


def load_recording(alert_id: str, as_of: datetime, runs_dir: Path | None = None) -> dict:
    """The recording for `alert_id`, or NoRecording. A recording made after `as_of` is never replayed (no future data)."""
    p = recording_path(alert_id, runs_dir)
    if not p.is_file():
        raise NoRecording(NO_RECORDING)
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec["version"] != RECORDING_VERSION or rec["alert_id"] != alert_id:
            raise ValueError("version or alert id mismatch")
        rec["model"], rec["provider"], rec["tool_calls"], rec["responses"]
        rec_as_of = parse_ts(rec["as_of"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise NoRecording(f"{NO_RECORDING} (the recording file is unusable: {exc})") from None
    if rec_as_of > as_of:
        raise NoRecording(
            f"{NO_RECORDING} at {iso_z(as_of)} (the recording was made at {rec['as_of']}, later than as_of)"
        )
    return rec


def _canon(args: dict) -> str:
    return json.dumps({k: v for k, v in args.items() if k != "as_of"}, sort_keys=True, default=str)


class ReplayLLM:
    """Returns the recorded model responses in order. No network, no client, no cost."""

    def __init__(self, rec: dict) -> None:
        self.queue = list(rec["responses"])
        self.model = rec["model"]

    def chat(self, role: str, messages: list[dict], tools=None, response_format=None) -> ChatResult:
        if not self.queue:
            raise ReplayMismatch("the recording has fewer model responses than this run asked for")
        r = self.queue.pop(0)
        return ChatResult(r["message"], r.get("usage") or {"input_tokens": 0, "output_tokens": 0}, 0, 0.0, self.model)


class ReplayTools:
    """Serves the recorded tool results in order and writes the same tool_log rows a live run writes."""

    def __init__(self, rec: dict) -> None:
        # calls the orchestrator itself rejected (bad arguments, disallowed tools) are re-created by it, not replayed
        self.queue = [t for t in rec["tool_calls"] if "_error" not in (t.get("args") or {})]

    def run(self, ctx: ToolContext, name: str, args: dict) -> tuple[dict, list[str]]:
        if not self.queue:
            raise ReplayMismatch(f"the recording has no result left for {name}")
        t = self.queue.pop(0)
        # the image prefetch asks for "the frame at as_of": its timestamp differs when the item was queued later than the
        # recording was made, so only the tray is compared (the recorded result names the frame that was used)
        same = (t["args"].get("tray_id") == args.get("tray_id")) if name == "get_frames" else _canon(t["args"]) == _canon(args)
        if t["tool"] != name or not same:
            raise ReplayMismatch(f"recorded call {t['tool']}({_canon(t['args'])}) does not match {name}({_canon(args)})")
        log_tool_call(ctx, name, json.loads(_canon(t["args"])), t["returned_ids"], t["result"])
        return t["result"], list(t["returned_ids"])

    def exhausted(self) -> bool:
        return not self.queue


class RecordingLLM:
    """Wraps a live client and keeps every response (message, usage, cost) for export_recording."""

    def __init__(self, llm: Any) -> None:
        self.llm, self.responses = llm, []

    def chat(self, role: str, messages: list[dict], tools=None, response_format=None) -> ChatResult:
        res = self.llm.chat(role, messages, tools=tools, response_format=response_format)
        self.responses.append({"message": res.message, "usage": res.usage, "cost_usd": res.cost_usd, "model": res.model})
        return res


def export_recording(conn: sqlite3.Connection, aid: str, recorder: RecordingLLM, mode: str) -> dict:
    """The recording of a finished live edge run (tool_log rows + model responses), as written to edge_runs/."""
    a = conn.execute("SELECT * FROM assessments WHERE id = ?", (aid,)).fetchone()
    calls = [
        {"round": r["round"], "tool": r["tool"], "args": json.loads(r["args"] or "{}"),
         "returned_ids": json.loads(r["returned_ids"] or "[]"), "result": json.loads(r["result"]) if r["result"] else None}
        for r in conn.execute("SELECT * FROM tool_log WHERE assessment_id = ? ORDER BY rowid", (aid,))
    ]
    return {
        "version": RECORDING_VERSION, "alert_id": a["alert_id"], "tray_id": a["tray_id"], "as_of": a["ts"],
        "assessment_id": aid, "mode": mode, "provider": a["provider"], "model": a["model"],
        "recorded_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "cost_usd": a["cost_usd"],
        "valid": bool(a["valid"]), "citation_errors": json.loads(a["citation_errors"] or "[]"),
        "tool_calls": calls, "responses": recorder.responses,
    }


# ---------------------------------------------------------------- the run
def _begin(conn: sqlite3.Connection, alert: dict, as_of: datetime, provider: str, model: str, recorded: bool) -> str:
    base = f"edge_{alert['tray_id']}_{iso_z(as_of).replace('-', '').replace(':', '')}"
    aid, k = base, 1
    while conn.execute("SELECT 1 FROM assessments WHERE id = ?", (aid,)).fetchone():
        k += 1
        aid = f"{base}_{k}"
    conn.execute(
        "INSERT INTO assessments(id, ts, tray_id, alert_id, model, provider, json, citation_errors, valid, latency_ms,"
        " cost_usd, recorded, updated_at, role) VALUES (?,?,?,?,?,?,NULL,NULL,NULL,NULL,0,?,?,'edge')",
        (aid, iso_z(as_of), alert["tray_id"], alert["id"], model, provider, 1 if recorded else 0, time.time()),
    )
    conn.commit()
    return aid


def _fail(conn: sqlite3.Connection, aid: str, msg: str, t0: float, cost: float) -> None:
    conn.execute(
        "UPDATE assessments SET json = NULL, citation_errors = ?, valid = 0, latency_ms = ?, cost_usd = ? WHERE id = ?",
        (json.dumps([f"error: {msg}"]), int((time.perf_counter() - t0) * 1000), cost, aid),
    )
    conn.commit()


def check_edge(conn: sqlite3.Connection, aid: str, edge: Any, alert: dict) -> tuple[list[str], list[dict] | None]:
    """(error strings, ungrounded numbers). Errors cover schema, citations, tray, numeric grounding and unbacked crew
    claims (the edge has no crew-notes tool, so every claim of a past crew action is unbacked). The second item is None
    when the numbers could not be evaluated (not a dict, or no stored tool results): "not evaluated", never "ok"."""
    errors = validate_edge(edge, aid, alert, conn)
    bad: list[dict] | None = None
    if isinstance(edge, dict):
        texts = grounding.edge_texts(edge)
        bad = grounding.check(conn, aid, alert, texts)
        errors += grounding.format_errors(bad or []) + grounding.crew_errors(texts, backed=False)
    return errors, bad


def _require_fit(messages: list[dict], limit: int | None = None) -> int:
    limit = MAX_PROMPT_TOKENS if limit is None else limit
    n = estimate_tokens(messages)
    if n > limit:
        raise TokenBudgetExceeded(f"edge prompt is ~{n} tokens, over the {limit}-token budget; call refused")
    return n


def triage(
    conn: sqlite3.Connection,
    alert_id: str,
    as_of: datetime,
    llm: Any | None = None,
    link: delay_sim.LinkState | None = None,
    runs_dir: Path | None = None,
    llm_factory: Callable[[sqlite3.Connection], Any] | None = None,
) -> str:
    """Triage one alert on board and persist the assessments row (role=edge). Returns its id.

    Raises AlertNotFound (agronomist.AlertNotFound) for an alert not raised at as_of and NoRecording in recorded mode
    without a usable recording; both before any row is written. A failed run is stored invalid, then re-raised."""
    mode = load_settings().edge_mode
    alert = find_alert(conn, alert_id, as_of)
    rec = load_recording(alert_id, as_of, runs_dir) if mode == "recorded" else None
    if rec is not None:
        provider, model = rec["provider"], rec["model"]
    else:
        cfg = resolve_role("edge")
        provider, model = cfg.provider, cfg.model
    aid = _begin(conn, alert, as_of, provider, model, rec is not None)
    t0 = time.perf_counter()
    try:
        if rec is not None:
            rec = json.loads(json.dumps(rec).replace(rec.get("assessment_id") or "\0", aid))  # calc ids carry the run id
            client, runner = ReplayLLM(rec), ReplayTools(rec)
        else:
            client, runner = llm or (llm_factory or LLMClient)(conn), None  # inside the try: a missing key is recorded
        _run(conn, aid, alert, as_of, client, runner, link or delay_sim.current(conn), t0)
        if runner is not None and (not runner.exhausted() or client.queue):
            raise ReplayMismatch("the recording has calls or responses this run did not use")
    except Exception as exc:
        log.warning("edge triage %s failed: %s", aid, exc)
        r = conn.execute("SELECT cost_usd FROM assessments WHERE id = ?", (aid,)).fetchone()
        _fail(conn, aid, f"{type(exc).__name__}: {exc}", t0, float(r["cost_usd"] or 0.0))
        raise
    return aid


def _run(
    conn: sqlite3.Connection, aid: str, alert: dict, as_of: datetime, llm: Any, runner: ReplayTools | None,
    link: delay_sim.LinkState, t0: float,
) -> None:
    replay = runner is not None
    trays = {t.id: t for t in load_trays().trays}
    tray_id = alert["tray_id"]
    focus = trays["A"] if tray_id == "air" else trays[tray_id]  # a shared air-sensor fault uses tray A as context
    ref = focus.reference_tray_id
    ctx = ToolContext(conn, aid, as_of, 0, alert["id"], alert["rule"])
    meter = _Meter(conn, aid, llm, "edge")

    def call(tool: str, args: dict, rnd: int) -> tuple[dict, list[str]]:
        ctx.round = rnd
        return runner.run(ctx, tool, args) if replay else run_tool(ctx, tool, args)

    overview, _ = call("get_tray_overview", {"tray_id": focus.id}, 0)
    frames, _ = call("get_frames", {"tray_id": focus.id, "timestamps": [iso_z(as_of)]}, 0)  # the one image
    refs = (frames.get("frame_refs") or [])[:1]
    part = None
    if refs and not replay:
        row = conn.execute("SELECT path FROM frames WHERE id = ?", (refs[0]["id"],)).fetchone()
        part = image_part(row["path"]) if row else None
        if part is None:  # unreadable file: the frame was never seen, so it is not citable
            set_logged_ids(ctx, ctx.last_log_rowid, [])
    has_image = bool(refs) and (part is not None or replay)
    text = _render(
        (PROMPTS / "edge.user.md").read_text(encoding="utf-8"),
        {
            "link_state": link.state, "delay_s": f"{link.delay_s:g}", "alert_id": alert["id"], "rule": alert["rule"],
            "tray_id": tray_id, "alert_ts": alert["ts"], "as_of": iso_z(as_of),
            "reference_desc": f"tray {ref}, same crop, control" if ref else f"tray {focus.id}'s own earlier baseline",
            "overview_alerted_json": json.dumps(overview, separators=(",", ":")),
        },
    )
    if not has_image:
        text = text.replace("[one tray image attached]", "[no tray image available]")
    user: dict = {"role": "user", "content": [{"type": "text", "text": text}, part] if part else text}
    messages: list[dict] = [
        {"role": "system", "content": (PROMPTS / "edge_sentinel.system.md").read_text(encoding="utf-8")}, user,
    ]
    _require_fit(messages)

    edge: Any = None
    errors: list[str] = []
    bad: list[dict] | None = []
    done = False
    for rnd in range(1, MAX_ROUNDS + 1):
        if estimate_tokens(messages) > MAX_PROMPT_TOKENS - FINAL_RESERVE_TOKENS:
            break  # no room for another tool round: go to the final answer
        res = meter.chat(messages, tools=EDGE_TOOL_SCHEMAS)
        messages.append(res.message)
        calls = res.message.get("tool_calls") or []
        if not calls:
            direct, perr = _parse_json(res.message.get("content") or "")
            if direct is not None and not perr:
                errs, _gb = check_edge(conn, aid, direct, alert)
                if not errs:
                    edge, errors, bad, done = direct, [], _gb, True
            break
        out: list[tuple[dict, dict]] = []
        for i, tc in enumerate(calls):
            fn = tc.get("function") or {}
            name = fn.get("name") or "?"
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise TypeError("arguments must be a JSON object")
                err = None
                if name not in EDGE_TOOLS:
                    err = f"tool {name!r} is not available on board; one of {list(EDGE_TOOLS)}"
                elif i >= MAX_CALLS_PER_ROUND:
                    err = f"at most {MAX_CALLS_PER_ROUND} tool calls per round"
            except (ValueError, TypeError) as e:
                args, err = {}, f"bad tool arguments: {e}"
            if err:
                ctx.round = rnd
                log_error_call(ctx, name, err)
                result = {"error": err}
            else:
                result, _ids = call(name, args, rnd)
                rowid = ctx.last_log_rowid
                body = json.dumps(result, separators=(",", ":"))
                probe = [*messages, {"role": "tool", "content": body}]
                if estimate_tokens(probe) > MAX_PROMPT_TOKENS - FINAL_RESERVE_TOKENS:
                    result = {"error": f"result omitted: the {MAX_PROMPT_TOKENS}-token prompt budget is reached"}
                    set_logged_ids(ctx, rowid, [])  # the model never saw it, so its ids are not citable
            out.append((tc, result))
        for tc, result in out:
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": json.dumps(result, separators=(",", ":"))})

    if not done:
        messages.append({"role": "user", "content": NUDGE})
        _require_fit(messages)
        res = meter.chat(messages, tools=None, response_format=response_format())
        edge, perr = _parse_json(res.message.get("content") or "")
        errors, bad = ([perr], None) if perr else check_edge(conn, aid, edge, alert)
        if errors:  # ONE repair call with the errors fed back
            messages.append(res.message)
            messages.append({
                "role": "user",
                "content": "Your answer failed validation:\n- " + "\n- ".join(errors)
                + "\nReturn the corrected full JSON. Tools are not available: cite only ids tools already returned (or the "
                "alert id) and use only numbers that appear in tool results.",
            })
            _require_fit(messages)
            res = meter.chat(messages, tools=None, response_format=response_format())
            retry, perr = _parse_json(res.message.get("content") or "")
            if perr:  # the FINAL reply is unparseable: nothing was evaluated, so there is no answer and no grounding result
                edge, errors, bad = None, [perr], None
            else:
                edge = retry
                errors, bad = check_edge(conn, aid, retry, alert)

    conn.execute(
        "UPDATE assessments SET json = ?, citation_errors = ?, valid = ?, latency_ms = ?, cost_usd = ?, warnings = ?"
        " WHERE id = ?",
        (json.dumps(edge) if edge is not None else None, json.dumps(errors), 0 if errors else 1,
         int((time.perf_counter() - t0) * 1000), meter.cost, json.dumps(bad) if bad is not None else None, aid),
    )
    conn.commit()


def edge_result(conn: sqlite3.Connection, aid: str) -> tuple[dict | None, sqlite3.Row]:
    a = conn.execute("SELECT * FROM assessments WHERE id = ?", (aid,)).fetchone()
    return (json.loads(a["json"]) if a["json"] else None), a



def edge_model_name(runs_dir: Path | None = None) -> str | None:
    """The model GET /link reports: the live model of the mode, or (recorded) the model inside the recordings, else None."""
    if load_settings().edge_mode == "recorded":
        models = recorded_models(runs_dir)
        return ", ".join(models) if models else None
    return resolve_role("edge").model


def recorded_alert_ids(runs_dir: Path | None = None) -> list[str]:
    return sorted(p.stem for p in Path(runs_dir or RUNS_DIR).glob("*.json"))
