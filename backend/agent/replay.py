"""Recorded Flight Agronomist runs for DEMO_MODE (M9). Same design as the edge recordings (backend/link/edge.py).

`data/demo/agent_runs/<alert_id>.json` holds every model response AND every tool result of one live run
(`scripts/record_demo.py`, paid, run by the lead). In DEMO_MODE `agronomist.begin/run` replay it through the SAME
orchestrator, so the replay writes the same `assessments` / `tool_log` / `actions` rows a live run writes, with
`recorded = 1`, the model that ACTUALLY answered, cost 0 and NO spend row (no LLMClient is ever built).

Nothing stored is trusted: the citation validator, the numeric grounding check and the action matching all run again on
the replayed answer, so a recording that no longer passes (changed rules, tools or prompts) is shown UNVERIFIED or
refused with ReplayMismatch. A missing recording raises NoAgentRecording; nothing falls back to a live call and nothing
is invented. A recording made at a mission time later than the requested `as_of` is never replayed (no future data).
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.agent.llm_client import ChatResult
from backend.agent.tools import ToolContext, log_tool_call, run_tool
from backend.config import ROOT
from backend.timeutil import iso_z, parse_ts

RUNS_DIR = ROOT / "data" / "demo" / "agent_runs"
RECORDING_VERSION = 1
NO_RECORDING = "no recorded run for this alert in demo mode"


class NoAgentRecording(LookupError):
    """DEMO_MODE and there is no (usable) recorded Flight Agronomist run for the alert."""


class ReplayMismatch(RuntimeError):
    """The recording does not fit the code path that replays it (changed tools or prompts): not replayed."""


def recording_path(alert_id: str, runs_dir: Path | None = None) -> Path:
    return Path(runs_dir or RUNS_DIR) / f"{alert_id}.json"


def recorded_alert_ids(runs_dir: Path | None = None) -> list[str]:
    return sorted(p.stem for p in Path(runs_dir or RUNS_DIR).glob("*.json"))


def load_recording(alert_id: str, as_of: datetime, runs_dir: Path | None = None) -> dict:
    """The recording for `alert_id`, or NoAgentRecording (missing, unusable, or made after `as_of`)."""
    p = recording_path(alert_id, runs_dir)
    if not p.is_file():
        raise NoAgentRecording(NO_RECORDING)
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec["version"] != RECORDING_VERSION or rec["alert_id"] != alert_id or rec.get("role") != "agronomist":
            raise ValueError("version, role or alert id mismatch")
        rec["model"], rec["provider"], rec["tool_calls"], rec["responses"]
        rec_as_of = parse_ts(rec["as_of"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise NoAgentRecording(f"{NO_RECORDING} (the recording file is unusable: {exc})") from None
    if rec_as_of > as_of:
        raise NoAgentRecording(
            f"{NO_RECORDING} at {iso_z(as_of)} (the recording was made at {rec['as_of']}, later than as_of)"
        )
    return rec


def rebind(rec: dict, aid: str) -> dict:
    """The recording with its original assessment id replaced by `aid` (calc ids carry the run id)."""
    old = rec.get("assessment_id")
    return json.loads(json.dumps(rec).replace(old, aid)) if old else rec


def _canon(args: dict) -> str:
    return json.dumps({k: v for k, v in args.items() if k != "as_of"}, sort_keys=True, default=str)


class ReplayLLM:
    """Returns the recorded model responses in order. No network, no client, no cost. `pace` is a pause per response."""

    def __init__(self, rec: dict, pace: float = 0.0, sleep=time.sleep) -> None:
        self.queue = list(rec["responses"])
        self.model, self.pace, self._sleep = rec["model"], pace, sleep

    def chat(self, role: str, messages: list[dict], tools=None, response_format=None) -> ChatResult:
        if not self.queue:
            raise ReplayMismatch("the recording has fewer model responses than this run asked for")
        if self.pace:
            self._sleep(self.pace)
        r = self.queue.pop(0)
        return ChatResult(r["message"], r.get("usage") or {"input_tokens": 0, "output_tokens": 0}, 0, 0.0, self.model)


class ReplayTools:
    """Serves the recorded tool results in order (checking the call matches) and logs them like a live run does.

    `propose_action` is the exception: it is re-run through the real handler, which re-checks the cited evidence ids
    against this run's log and writes the `actions` row."""

    def __init__(self, rec: dict, pace: float = 0.0, sleep=time.sleep) -> None:
        # calls the orchestrator itself rejected (bad arguments, over the per-round limit) are re-created by it
        self.queue = [t for t in rec["tool_calls"] if "_error" not in (t.get("args") or {})]
        self.pace, self._sleep = pace, sleep

    def run(self, ctx: ToolContext, name: str, args: dict) -> tuple[dict, list[str]]:
        if not self.queue:
            raise ReplayMismatch(f"the recording has no result left for {name}")
        t = self.queue.pop(0)
        if t["tool"] != name or _canon(t["args"]) != _canon(args):
            raise ReplayMismatch(f"recorded call {t['tool']}({_canon(t['args'])}) does not match {name}({_canon(args)})")
        if self.pace:
            self._sleep(self.pace)
        if name == "propose_action":
            return run_tool(ctx, name, json.loads(_canon(t["args"])))
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


def export_recording(conn: sqlite3.Connection, aid: str, recorder: RecordingLLM) -> dict:
    """The recording of a finished live run (tool_log rows + model responses), as written to agent_runs/."""
    a = conn.execute("SELECT * FROM assessments WHERE id = ?", (aid,)).fetchone()
    calls = [
        {"round": r["round"], "tool": r["tool"], "args": json.loads(r["args"] or "{}"),
         "returned_ids": json.loads(r["returned_ids"] or "[]"), "result": json.loads(r["result"]) if r["result"] else None}
        for r in conn.execute("SELECT * FROM tool_log WHERE assessment_id = ? ORDER BY rowid", (aid,))
    ]
    return {
        "version": RECORDING_VERSION, "role": "agronomist", "alert_id": a["alert_id"], "tray_id": a["tray_id"],
        "as_of": a["ts"], "assessment_id": aid, "provider": a["provider"], "model": a["model"],
        "recorded_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "cost_usd": a["cost_usd"],
        "valid": bool(a["valid"]), "citation_errors": json.loads(a["citation_errors"] or "[]"),
        "tool_calls": calls, "responses": recorder.responses,
    }
