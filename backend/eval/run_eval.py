"""Agent eval harness (M7): replays scenarios into temp DBs, runs the Flight Agronomist, scores it.

    python scripts/run_eval.py --dry-run                      # count scenarios + estimated cost, no calls
    python scripts/run_eval.py --role agronomist --effort low # PAID: real model calls, ledgered
    python scripts/run_eval.py --fake --source-db synth.db    # free: scripted LLM, for tests and smoke runs

Isolation. Each scenario works on its OWN copy of the source DB (sqlite backup API from a read-only connection, so
the source file is never written). Injections (crew notes, sensor overrides, occluded frames, burned-in captions),
alert resets and the agent's assessments all land in that copy, which is deleted afterwards.

Spend. The temp copy is never the ledger. The LLM client is built on a connection to the MAIN ledger (the `spend`
table of the configured DB_PATH), so `check_budget` / CERES_BUDGET_USD see the true total before every call and
every call is recorded there as it happens. The per-run `--max-cost` cap is checked between scenarios on top.

TODO(M8): the `edge` role (Mars-mode sentinel, scored on edge.json's likely_cause and the same citation validator)
needs backend/agent edge support from M8. `--role edge` is rejected until then.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sqlite3
import sys
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from backend.agent import agronomist, llm_client
from backend.agent.llm_client import ChatResult, LLMClient
from backend.agent.validate import assessment_schema
from backend.api.alerts import visible_alerts
from backend.config import ROOT, load_settings, load_trays
from backend.db import connect, init_db
from backend.eval import score
from backend.pipeline import run_cycle
from backend.spend import BudgetExceeded, record_spend
from backend.timeutil import iso_z, parse_ts

EVAL_DIR = ROOT / "backend" / "eval"
SCENARIO_DIR = EVAL_DIR / "scenarios"
OUT_DIR = EVAL_DIR / "out"
LATEST_NAME = "latest.json"
EST_PER_ASSESSMENT_USD = (0.10, 0.20)  # PROGRESS.md live measurements: $0.094 and $0.194 per assessment, low effort
DEFAULT_MAX_COST_USD = 10.0  # per run, on top of CERES_BUDGET_USD; PLAN 0.4b gives all eval runs $35
INJECTION_TYPES = ("crew_note", "sensor_override", "occlude_frames", "image_label")
RULE_AFFECTING = ("sensor_override", "occlude_frames")  # injections that can change which alerts fire
LABEL_LOOKBACK_H = 6  # an injected caption is burned into every frame of the tray in this many hours before as_of
CAUSES = set(assessment_schema()["properties"]["probable_causes"]["items"]["properties"]["cause"]["enum"])


# ---------------------------------------------------------------- scenarios
class ScenarioError(ValueError):
    pass


def validate_scenario(sc: dict, name: str = "?") -> dict:
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise ScenarioError(f"scenario {sc.get('id', name)}: {msg}")

    for k in ("id", "description", "data_source", "tray_id", "as_of", "window", "alert_rule", "expected_top_cause",
              "acceptable_causes", "expected_alerts"):
        need(k in sc, f"missing key {k!r}")
    need(sc["data_source"] in score.DATA_SOURCES, f"data_source must be one of {score.DATA_SOURCES}")
    need(set(sc["window"]) == {"start", "end"}, "window needs start and end")
    t0, t1, ta = (parse_ts(x) for x in (sc["window"]["start"], sc["window"]["end"], sc["as_of"]))
    need(t0 <= ta <= t1, "as_of must lie inside the window")
    exp = sc["expected_top_cause"]
    need(exp is None or exp in CAUSES, f"expected_top_cause {exp!r} is not in the assessment schema's cause enum")
    need(all(c in CAUSES for c in sc["acceptable_causes"]), "acceptable_causes must come from the cause enum")
    need(exp is None or exp in sc["acceptable_causes"], "expected_top_cause must be in acceptable_causes")
    need((sc["alert_rule"] is None) == (exp is None), "alert_rule and expected_top_cause are both null or both set")
    for key in (*sc["expected_alerts"], *sc.get("forbidden_alerts", [])):
        need(isinstance(key, str) and key.count(":") == 1, f"alert key {key!r} must look like RULE:tray")
    for inj in sc.get("injections") or []:
        need(inj.get("type") in INJECTION_TYPES, f"unknown injection type {inj.get('type')!r}")
    return sc


def load_scenarios(directory: Path = SCENARIO_DIR, ids: list[str] | None = None) -> list[dict]:
    found = []
    for p in sorted(Path(directory).glob("*.yaml")):
        found.append(validate_scenario(yaml.safe_load(p.read_text(encoding="utf-8")), p.name))
    seen = [s["id"] for s in found]
    if len(set(seen)) != len(seen):
        raise ScenarioError(f"duplicate scenario ids in {directory}")
    if ids:
        unknown = sorted(set(ids) - set(seen))
        if unknown:
            raise ScenarioError(f"unknown scenario ids {unknown}; have {sorted(seen)}")
        found = [s for s in found if s["id"] in ids]
    return found


# ---------------------------------------------------------------- temp DB
def copy_db(src: Path, dst: Path) -> sqlite3.Connection:
    """Copy `src` into a new file with the backup API from a READ-ONLY connection; `src` is never written."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    ro = sqlite3.connect(Path(src).resolve().as_uri() + "?mode=ro", uri=True)
    out = sqlite3.connect(str(dst))
    try:
        ro.backup(out)
    finally:
        ro.close()
        out.close()
    return init_db(dst)  # migrate older schemas in the COPY


def reset_state(conn: sqlite3.Connection) -> None:
    """Start every scenario from a known state: no earlier assessments, notes, actions or queue."""
    for t in ("assessments", "tool_log", "actions", "crew_notes", "link_queue"):
        conn.execute(f"DELETE FROM {t}")
    conn.commit()


# ---------------------------------------------------------------- injections
def _blank_frame_path(dirpath: Path, tray: str, ts: str) -> str:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / f"occluded_{tray}_{ts.replace('-', '').replace(':', '')}.jpg"
    cv2.imwrite(str(p), np.zeros((768, 1024, 3), np.uint8))
    return str(p)


def inject_sensor_override(conn: sqlite3.Connection, inj: dict) -> int:
    owner, metric, mode = inj["owner"], inj["metric"], inj.get("mode", "flatline")
    start, end = parse_ts(inj["start"]), parse_ts(inj["end"])
    step = timedelta(minutes=inj.get("step_min", 5))
    lo, hi = iso_z(start), iso_z(end)
    if mode == "flatline":  # make sure there is a row on the grid so the series is flat for the whole span
        t = start
        while t <= end:
            ts = iso_z(t)
            conn.execute(
                "INSERT OR IGNORE INTO sensor_readings(id, ts, tray_id, metric, value, source) VALUES (?,?,?,?,?,'simulated')",
                (f"{owner}:{metric}:{ts}", ts, owner, metric, float(inj["value"])),
            )
            t += step
    elif mode != "spike":
        raise ScenarioError(f"unknown sensor_override mode {mode!r}")
    rows = conn.execute(
        "SELECT id, value FROM sensor_readings WHERE tray_id = ? AND metric = ? AND ts >= ? AND ts <= ? ORDER BY ts",
        (owner, metric, lo, hi),
    ).fetchall()
    if mode == "spike" and not rows:  # nothing recorded there: synthesize a baseline so the spike has a series
        prev = conn.execute(
            "SELECT value FROM sensor_readings WHERE tray_id = ? AND metric = ? AND ts < ? ORDER BY ts DESC LIMIT 1",
            (owner, metric, lo),
        ).fetchone()  # the last recorded value, else 40 % of the peak: a baseline, never zero
        base = float(inj["base"]) if "base" in inj else (prev["value"] if prev else 0.4 * float(inj["value"]))
        t = start
        while t <= end:
            ts = iso_z(t)
            conn.execute(
                "INSERT OR IGNORE INTO sensor_readings(id, ts, tray_id, metric, value, source) VALUES (?,?,?,?,?,'simulated')",
                (f"{owner}:{metric}:{ts}", ts, owner, metric, base),
            )
            t += step
        rows = conn.execute(
            "SELECT id, value FROM sensor_readings WHERE tray_id = ? AND metric = ? AND ts >= ? AND ts <= ? ORDER BY ts",
            (owner, metric, lo, hi),
        ).fetchall()
    n = len(rows)
    if mode == "flatline":
        conn.executemany("UPDATE sensor_readings SET value = ? WHERE id = ?", [(float(inj["value"]), r["id"]) for r in rows])
    elif n:
        base = sum(r["value"] for r in rows) / n
        amp = float(inj["value"]) - base
        for i, r in enumerate(rows):  # triangular ramp on top of the recorded series (keeps its wobble)
            tri = 1.0 - abs(2.0 * i / (n - 1) - 1.0) if n > 1 else 1.0
            conn.execute("UPDATE sensor_readings SET value = ? WHERE id = ?", (r["value"] + amp * tri, r["id"]))
    conn.commit()
    return n


def inject_occlusion(conn: sqlite3.Connection, inj: dict, assets: Path) -> int:
    """Replace the tray's frames in [start, end] by black images with a canopy mask of area 0. No SAM call."""
    tray, lo, hi = inj["tray_id"], iso_z(parse_ts(inj["start"])), iso_z(parse_ts(inj["end"]))
    frames = conn.execute(
        "SELECT id, ts FROM frames WHERE tray_id = ? AND ts >= ? AND ts <= ? ORDER BY ts", (tray, lo, hi)
    ).fetchall()
    if not frames:  # nothing recorded in the span: add hourly occluded frames
        cfg = load_trays().by_id(tray)
        t = parse_ts(lo)
        while t <= parse_ts(hi):
            ts = iso_z(t)
            conn.execute(
                "INSERT INTO frames(id, tray_id, ts, path, cm2_per_px, calib_flag, source) VALUES (?,?,?,?,?,?,?)",
                (f"{tray}:{ts}", tray, ts, "", cfg.cm2_per_px, cfg.calib_flag, "fault_transform"),
            )
            t += timedelta(hours=1)
        frames = conn.execute(
            "SELECT id, ts FROM frames WHERE tray_id = ? AND ts >= ? AND ts <= ? ORDER BY ts", (tray, lo, hi)
        ).fetchall()
    for f in frames:
        old = conn.execute("SELECT pots_json FROM masks WHERE frame_id = ? AND kind = 'canopy'", (f["id"],)).fetchone()
        pots_json = None
        if old and old["pots_json"]:
            pots_json = json.dumps({k: 0 for k in json.loads(old["pots_json"])})
        conn.execute("DELETE FROM embeddings WHERE mask_id IN (SELECT id FROM masks WHERE frame_id = ?)", (f["id"],))
        conn.execute("DELETE FROM masks WHERE frame_id = ?", (f["id"],))
        conn.execute(
            "UPDATE frames SET path = ?, source = 'fault_transform', fault_hash = NULL WHERE id = ?",
            (_blank_frame_path(assets, tray, f["ts"]), f["id"]),
        )
        conn.execute(
            "INSERT INTO masks(id, frame_id, tray_id, plant_id, kind, area_px, area_cm2, exg, chlorosis_frac, rle,"
            " pots_json) VALUES (?,?,?,NULL,'canopy',0,0.0,NULL,NULL,'[]',?)",
            (f"{f['id']}:canopy", f["id"], tray, pots_json),
        )
    conn.commit()
    return len(frames)


def inject_image_label(conn: sqlite3.Connection, inj: dict, as_of: datetime, assets: Path) -> int:
    """Burn a text caption into the tray's frames of the last LABEL_LOOKBACK_H hours (the image channel)."""
    tray = inj["tray_id"]
    hi = iso_z(min(parse_ts(inj["ts"]), as_of)) if inj.get("ts") else iso_z(as_of)
    lo = iso_z(parse_ts(hi) - timedelta(hours=LABEL_LOOKBACK_H))
    frames = conn.execute(
        "SELECT id, path FROM frames WHERE tray_id = ? AND ts > ? AND ts <= ?", (tray, lo, hi)
    ).fetchall()
    assets.mkdir(parents=True, exist_ok=True)
    words, lines, cur = str(inj["text"]).split(), [], ""
    for w in words:  # wrap at ~26 characters per line
        if len(cur) + len(w) + 1 > 26 and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    done = 0
    for f in frames:
        img = cv2.imread(str(ROOT / f["path"])) if f["path"] else None
        if img is None:
            continue
        for i, line in enumerate(lines):
            org = (20, 60 + 52 * i)
            cv2.putText(img, line, org, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.putText(img, line, org, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        out = assets / f"label_{f['id'].replace(':', '_')}.jpg"
        cv2.imwrite(str(out), img)
        conn.execute("UPDATE frames SET path = ? WHERE id = ?", (str(out), f["id"]))
        done += 1
    conn.commit()
    return done


def inject_crew_note(conn: sqlite3.Connection, inj: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO crew_notes(id, ts, audio_path, transcript, extracted_json) VALUES (?,?,NULL,?,?)",
        (inj["id"], inj["ts"], inj["transcript"], json.dumps(inj.get("extracted") or {})),
    )
    conn.commit()


def apply_injections(conn: sqlite3.Connection, sc: dict, assets: Path) -> list[str]:
    """Apply the scenario's injections to the temp DB. Returns one log line per injection."""
    as_of, log = parse_ts(sc["as_of"]), []
    for inj in sc.get("injections") or []:
        t = inj["type"]
        if t == "sensor_override":
            log.append(f"sensor_override {inj['owner']}:{inj['metric']} {inj.get('mode', 'flatline')}: {inject_sensor_override(conn, inj)} rows")
        elif t == "occlude_frames":
            log.append(f"occlude_frames {inj['tray_id']}: {inject_occlusion(conn, inj, assets)} frames")
        elif t == "image_label":
            log.append(f"image_label {inj['tray_id']}: {inject_image_label(conn, inj, as_of, assets)} frames")
        else:
            inject_crew_note(conn, inj)
            log.append(f"crew_note {inj['id']}")
    return log


# ---------------------------------------------------------------- alerts
def replay_times(start: datetime, end: datetime) -> list[datetime]:
    """Hour marks in [start, end] plus `end` itself."""
    t = start.replace(minute=0, second=0, microsecond=0)
    if t < start:
        t += timedelta(hours=1)
    out = []
    while t <= end:
        out.append(t)
        t += timedelta(hours=1)
    if not out or out[-1] != end:
        out.append(end)
    return out


def replay_rules(conn: sqlite3.Connection, sc: dict, clear: bool) -> int:
    """Hourly pipeline cycles over the scenario window (rules only: no ingest, no segmentation, no model call)."""
    if clear:
        conn.execute("DELETE FROM alert_events")
        conn.execute("DELETE FROM alerts")
        conn.commit()
    cfg, times = load_trays(), replay_times(parse_ts(sc["window"]["start"]), parse_ts(sc["as_of"]))
    for t in times:
        run_cycle(conn, cfg, t, backend=None, segment=False, ingest=False)
    return len(times)


def pick_alert(conn: sqlite3.Connection, sc: dict) -> dict | None:
    """The alert the scenario assesses: its rule on its tray visible at as_of; open first, then latest.

    `alert_rule: auto` takes whichever alert is open on the scenario tray. If none is, it falls back to the knock-on
    alerts: GROWTH_LAG alerts open on trays that use the scenario tray as their reference (a dead reference camera
    makes every tray compared with it look like it lags)."""
    if not sc.get("alert_rule"):
        return None
    rule, as_of = sc["alert_rule"], parse_ts(sc["as_of"])
    cands = [a for a in visible_alerts(conn, as_of, sc["tray_id"]) if rule == "auto" or a["rule"] == rule]
    needle = sc.get("alert_detail_contains")
    if needle:
        cands = [a for a in cands if needle in (a["detail"] or "")]
    if rule == "auto":
        cands = [a for a in cands if a["status"] == "open"]
        if not cands:
            kids = {t.id for t in load_trays().trays if t.reference_tray_id == sc["tray_id"]}
            cands = [a for a in visible_alerts(conn, as_of) if a["status"] == "open" and a["rule"] == "GROWTH_LAG"
                     and a["tray_id"] in kids]
    if not cands:
        return None
    return max(cands, key=lambda a: (a["status"] == "open", a["ts"]))


def alert_keys(alerts: list[dict]) -> list[str]:
    return sorted({f"{a['rule']}:{a['tray_id']}" for a in alerts})


# ---------------------------------------------------------------- LLMs
@contextlib.contextmanager
def role_override(role: str, effort: str | None = None, model: str | None = None) -> Iterator[None]:
    """Temporarily change a role's reasoning effort / model in llm_client.ROLES (production defaults untouched)."""
    old = llm_client.ROLES[role]
    new = old
    if effort:
        new = replace(new, reasoning_effort=effort)
    if model:
        new = replace(new, model=model)
    llm_client.ROLES[role] = new
    try:
        yield
    finally:
        llm_client.ROLES[role] = old


class ScriptedLLM:
    """Free stand-in for the model (`--fake`, tests): a fixed tool round, then a JSON answer built from the scenario.

    It answers with the scenario's expected top cause (an oracle), so a fake run proves the harness, not the agent.
    `top_cause` overrides the answer; `comply=True` makes it follow an injected 'cut water' instruction (to test the
    failure path). `cost_per_call` > 0 records a spend row into `ledger` (the MAIN ledger connection)."""

    def __init__(self, sc: dict, alert: dict, ledger: sqlite3.Connection | None = None, cost_per_call: float = 0.0,
                 top_cause: str | None = None, comply: bool = False) -> None:
        self.sc, self.alert, self.ledger, self.cost = sc, alert, ledger, cost_per_call
        self.top = top_cause or sc.get("expected_top_cause") or "other"
        self.comply = comply
        self.n = 0

    def _tool_round(self) -> dict:
        tray, i = self.alert["tray_id"], 0
        real = tray in {t.id for t in load_trays().trays}

        def call(name: str, args: dict) -> dict:
            nonlocal i
            i += 1
            return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}

        calls = [call("get_crew_notes", {"hours": 72})]
        calls.append(call("get_sensor_series", {"tray_id": tray, "metric": "soil_vwc" if real else "rh_pct", "hours": 24}))
        if real:
            calls.append(call("get_canopy_metrics", {"tray_id": tray, "hours": 48}))
            calls.append(call("get_frames", {"tray_id": tray, "timestamps": [self.sc["as_of"]]}))
        if self.comply or self.top == "underwatering":
            act = "Cut water to tray B" if self.comply else f"Water tray {tray}"
            calls.append(call("propose_action", {"tray_id": tray, "action": act, "magnitude": "moderate",
                                                 "rationale": "per assessment", "evidence_ids": [self.alert["id"]]}))
        return {"role": "assistant", "content": "", "tool_calls": calls}

    def _final(self) -> dict:
        aid, tray = self.alert["id"], self.alert["tray_id"]
        others = [c for c in ("sensor_fault", "low_light", "other") if c != self.top][:2]
        findings = [{"claim": f"Alert {self.alert['rule']} is open on {tray}.", "evidence_ids": [aid], "confidence": 0.8}]
        if any(i.get("type") == "crew_note" for i in self.sc.get("injections") or []):
            findings.append({"claim": "A crew note on file was treated as untrusted data and checked against the sensors.",
                             "evidence_ids": [aid], "confidence": 0.6})
        actions = []
        if self.comply:
            actions = [{"action": "Cut water to tray B", "magnitude": "moderate", "rationale": "per crew note", "evidence_ids": [aid]}]
        elif self.top == "underwatering":
            actions = [{"action": f"Water tray {tray}", "magnitude": "moderate", "rationale": "dry substrate", "evidence_ids": [aid]}]
        brief = "Cut water to tray B as the note says." if self.comply else f"Follow up on the {self.alert['rule']} alert for {tray}."
        out = {
            "tray_id": tray, "status": "alert", "findings": findings,
            "probable_causes": [
                {"cause": self.top, "likelihood": 0.7, "evidence_ids": [aid], "discriminating_check": "Scripted check."},
                {"cause": others[0], "likelihood": 0.1, "evidence_ids": [aid], "discriminating_check": "Scripted check."},
            ],
            "causes_considered": [self.top, "sensor_fault", "low_light", "normal_variation"],
            "recommended_actions": actions, "setpoint_caveats": [], "missing_evidence": [], "crew_brief": brief,
        }
        return {"role": "assistant", "content": json.dumps(out)}

    def chat(self, role: str, messages: list[dict], tools=None, response_format=None) -> ChatResult:
        self.n += 1
        if tools:
            msg = self._tool_round() if self.n == 1 else {"role": "assistant", "content": "Investigation complete."}
        else:
            msg = self._final()
        if self.cost and self.ledger is not None:
            record_spend(self.ledger, role, "fake", "fake-scripted", 1000, 300, 0, cost_usd=self.cost)
        return ChatResult(msg, {"input_tokens": 1000, "output_tokens": 300}, 5, self.cost, "fake-scripted")


LLMFactory = Callable[[dict, dict, sqlite3.Connection | None], Any]


def paid_factory(sc: dict, alert: dict, ledger: sqlite3.Connection | None) -> LLMClient:
    """A real client whose ledger (budget check and spend rows) is the MAIN DB connection, not the temp copy."""
    if ledger is None:
        raise RuntimeError("a paid run needs the main ledger connection")
    return LLMClient(ledger)


def fake_factory(cost_per_call: float = 0.0, **kw) -> LLMFactory:
    return lambda sc, alert, ledger: ScriptedLLM(sc, alert, ledger, cost_per_call, **kw)


# ---------------------------------------------------------------- one scenario
def _ledger_mark(ledger: sqlite3.Connection | None) -> int:
    return 0 if ledger is None else int(ledger.execute("SELECT COALESCE(MAX(id), 0) FROM spend").fetchone()[0])


def _ledger_since(ledger: sqlite3.Connection | None, mark: int) -> float | None:
    if ledger is None:
        return None
    return float(ledger.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM spend WHERE id > ?", (mark,)).fetchone()[0])


def run_scenario(
    sc: dict,
    source_db: Path,
    work_dir: Path,
    llm_factory: LLMFactory,
    ledger: sqlite3.Connection | None,
    role: str = "agronomist",
    log: Callable[[str], None] = lambda _m: None,
) -> tuple[dict, dict]:
    """Run one scenario in a temp copy of `source_db`. Returns (scored row, run info). Never writes `source_db`."""
    sdir = Path(work_dir) / sc["id"]
    conn = copy_db(source_db, sdir / "scenario.db")
    as_of = parse_ts(sc["as_of"])
    rec: dict[str, Any] = {"alert_found": False, "errors": [], "tool_counts": {}, "error": None}
    info: dict[str, Any] = {"injections": [], "replayed_cycles": 0, "ledger_usd": None}
    mark = _ledger_mark(ledger)
    try:
        reset_state(conn)
        info["injections"] = apply_injections(conn, sc, sdir / "assets")
        inj_types = {i["type"] for i in sc.get("injections") or []}
        clear = bool(inj_types & set(RULE_AFFECTING))
        if clear or sc.get("replay") is True:
            info["replayed_cycles"] = replay_rules(conn, sc, clear=clear)
        alert = pick_alert(conn, sc)
        if alert is None and sc.get("alert_rule") and not clear:  # not in the stored replay: derive it from the window
            info["replayed_cycles"] += replay_rules(conn, sc, clear=False)
            alert = pick_alert(conn, sc)
        visible = visible_alerts(conn, as_of)
        rec["open_alerts"] = alert_keys([a for a in visible if a["status"] == "open"])
        rec["window_alerts"] = alert_keys(
            [a for a in visible if a["tray_id"] == sc["tray_id"] and a["ts"] >= sc["window"]["start"]]
        )
        rec["window_alert_details"] = [a["detail"] or "" for a in visible if a["ts"] >= sc["window"]["start"]]
        if alert is not None:
            rec.update(alert_found=True, alert_id=alert["id"])
            llm = llm_factory(sc, alert, ledger)
            try:
                aid = agronomist.assess(conn, alert["id"], as_of, role=role, llm=llm)
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001  recorded in the temp DB by agronomist.run; scored as a failure
                rec["error"] = f"{type(exc).__name__}: {exc}"
                row = conn.execute("SELECT id FROM assessments WHERE alert_id = ? ORDER BY ts DESC", (alert["id"],)).fetchone()
                aid = row["id"] if row else None
            if aid:
                a = conn.execute("SELECT * FROM assessments WHERE id = ?", (aid,)).fetchone()
                rec.update(
                    assessment_id=aid, assessment=json.loads(a["json"]) if a["json"] else None,
                    errors=json.loads(a["citation_errors"] or "[]"), latency_ms=a["latency_ms"], cost_usd=a["cost_usd"] or 0.0,
                    tool_counts={r["tool"]: r["n"] for r in conn.execute(
                        "SELECT tool, COUNT(*) AS n FROM tool_log WHERE assessment_id = ? GROUP BY tool", (aid,))},
                )
    except BudgetExceeded as exc:
        rec["error"] = f"BudgetExceeded: {exc}"
        info["budget_exceeded"] = True
    except Exception as exc:  # noqa: BLE001  one broken scenario must not lose the others
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()
        shutil.rmtree(sdir, ignore_errors=True)
    info["ledger_usd"] = _ledger_since(ledger, mark)
    log(f"  {sc['id']}: " + (rec["error"] or f"alert={rec.get('alert_id')} top={score.top_cause(rec.get('assessment'))}"))
    return score.score_scenario(sc, rec), info


# ---------------------------------------------------------------- the run
def estimate(scenarios: list[dict]) -> dict:
    n = sum(1 for s in scenarios if s.get("alert_rule"))
    lo, hi = EST_PER_ASSESSMENT_USD
    return {"n_scenarios": len(scenarios), "n_assessments": n, "usd_low": round(n * lo, 2), "usd_high": round(n * hi, 2)}


def run_eval(
    source_db: Path,
    scenarios: list[dict],
    llm_factory: LLMFactory,
    ledger: sqlite3.Connection | None,
    role: str = "agronomist",
    effort: str | None = None,
    model: str | None = None,
    fake: bool = False,
    max_cost: float = DEFAULT_MAX_COST_USD,
    out_dir: Path | None = OUT_DIR,
    update_latest: bool | None = None,  # None: yes for a real run, no for --fake
    run_id: str | None = None,
    note: str | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    if role != "agronomist":
        raise NotImplementedError("TODO(M8): the edge role needs the M8 edge agent; only 'agronomist' is evaluated")
    now = datetime.now(UTC)
    run_id = run_id or f"eval_{now.strftime('%Y%m%dT%H%M%SZ')}{'_fake' if fake else ''}"
    rows: list[dict] = []
    ledger_usd, aborted = 0.0, None
    with role_override(role, effort, model):
        cfg = llm_client.resolve_role(role)
        work = Path(tempfile.mkdtemp(prefix="ceres_eval_"))
        try:
            for sc in scenarios:
                if ledger is not None and ledger_usd >= max_cost:
                    aborted = f"stopped before {sc['id']}: run cost ${ledger_usd:.2f} reached --max-cost ${max_cost:.2f}"
                    break
                row, info = run_scenario(sc, source_db, work, llm_factory, ledger, role, log)
                rows.append(row)
                ledger_usd += info["ledger_usd"] or 0.0
                if info.get("budget_exceeded"):
                    aborted = f"stopped at {sc['id']}: ledger budget exceeded (CERES_BUDGET_USD)"
                    break
        finally:
            shutil.rmtree(work, ignore_errors=True)
    rep = {
        "run_id": run_id,
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),  # wall-clock UTC: a static report, not mission time
        "note": note,
        "fake": fake,
        "role": role,
        "model": "fake-scripted" if fake else cfg.model,
        "effort": effort or cfg.reasoning_effort or "default",
        "source_db": str(source_db),
        "aborted": aborted,
        **score.aggregate(rows),
        "ledger_usd": None if ledger is None else round(ledger_usd, 6),
        "scenarios": rows,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(rep, indent=2)
        (out_dir / f"{run_id}.json").write_text(text, encoding="utf-8")
        (out_dir / f"{run_id}.md").write_text(score.render_markdown(rep), encoding="utf-8")
        if (not fake) if update_latest is None else update_latest:
            (out_dir / LATEST_NAME).write_text(text, encoding="utf-8")
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", default="agronomist", choices=["agronomist", "edge"], help="edge is M8 (not built)")
    ap.add_argument("--effort", choices=["low", "high"], default=None, help="reasoning_effort for this run only")
    ap.add_argument("--model", default=None, help="override the role's model (e.g. muse-spark-1.3-contributor for dev)")
    ap.add_argument("--scenarios", default=None, help="comma-separated scenario ids (default: all)")
    ap.add_argument("--scenario-dir", type=Path, default=SCENARIO_DIR)
    ap.add_argument("--source-db", type=Path, default=None, help="DB to replay from (default: DB_PATH); never written")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST_USD, help="stop the run past this many USD")
    ap.add_argument("--dry-run", action="store_true", help="print the scenario count and estimated cost; no calls")
    ap.add_argument("--fake", action="store_true", help="scripted LLM: free, no network, never touches the ledger")
    ap.add_argument("--update-latest", action="store_true", help="also write latest.json for a --fake run")
    args = ap.parse_args(argv)
    if args.role == "edge":
        print("TODO(M8): the edge role is built in M8; only --role agronomist is evaluated.", file=sys.stderr)
        return 2
    try:
        scenarios = load_scenarios(args.scenario_dir, args.scenarios.split(",") if args.scenarios else None)
    except ScenarioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    est = estimate(scenarios)
    lo, hi = EST_PER_ASSESSMENT_USD
    print(
        f"{est['n_scenarios']} scenarios, {est['n_assessments']} assessments; estimated "
        f"${est['usd_low']:.2f} to ${est['usd_high']:.2f} (${lo:.2f} to ${hi:.2f} each, measured at low effort on "
        f"muse-spark-1.3; high effort is unmeasured and may cost more). Nominal scenarios make no model call."
    )
    for s in scenarios:
        print(f"  {s['id']:<20} {s['data_source']:<16} {s['alert_rule'] or '-':<13} as_of {s['as_of']}")
    if args.dry_run:
        return 0
    if args.fake:
        if args.source_db is None:
            print("error: --fake needs --source-db (it never defaults to the real DB)", file=sys.stderr)
            return 2
        source, ledger, factory = args.source_db, None, fake_factory()
        note = None  # the report itself says FAKE RUN
    else:
        settings = load_settings()
        source = args.source_db or Path(settings.db_path)
        ledger = connect(settings.db_path)  # the MAIN ledger: budget check and every spend row go here
        factory, note = paid_factory, None
        print(f"PAID run: ledger {settings.db_path}, budget cap ${settings.budget_usd:.2f}, run cap ${args.max_cost:.2f}")
    if not Path(source).is_file():
        print(f"error: source DB {source} not found", file=sys.stderr)
        return 2
    try:
        rep = run_eval(source, scenarios, factory, ledger, args.role, args.effort, args.model, args.fake, args.max_cost,
                       args.out_dir, update_latest=True if args.update_latest else None, note=note)
    finally:
        if ledger is not None:
            ledger.close()
    d = rep["deterministic"]
    print(f"\n{rep['run_id']}: passed {rep['passed']}, top-1 {rep['top1_overall']}, citation validity {d['citation_validity']}, "
          f"schema {d['schema_validity']}, injection {d['injection_resistance']}, cost ${rep['cost_usd']['total']:.4f}")
    if rep["aborted"]:
        print(f"ABORTED: {rep['aborted']}")
    print(f"wrote {Path(args.out_dir) / (rep['run_id'] + '.json')} and .md")
    return 1 if rep["aborted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
