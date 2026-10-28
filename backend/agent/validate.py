"""Deterministic validation of an assessment against its tool_log (M4b).

JSON Schema cannot express citation windows, the likelihood sum or "sensor_fault must be considered", and a
provider's strict mode may drop schema keywords, so everything is re-checked here in code.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator

from backend.api.alerts import visible_alerts
from backend.timeutil import parse_ts

SCHEMA_PATH = Path(__file__).with_name("schemas") / "assessment.json"


@lru_cache(maxsize=1)
def assessment_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def logged_ids(conn: sqlite3.Connection, assessment_id: str) -> set[str]:
    out: set[str] = set()
    for r in conn.execute("SELECT returned_ids FROM tool_log WHERE assessment_id = ?", (assessment_id,)):
        out.update(json.loads(r["returned_ids"] or "[]"))
    return out


def _as_of(conn: sqlite3.Connection, assessment_id: str, alert: dict, as_of: datetime | None) -> datetime:
    if as_of is not None:
        return as_of
    r = conn.execute("SELECT ts FROM assessments WHERE id = ?", (assessment_id,)).fetchone()
    return parse_ts(r["ts"] if r and r["ts"] else alert["ts"])


def validate(
    assessment: dict, assessment_id: str, alert: dict, conn: sqlite3.Connection, as_of: datetime | None = None
) -> list[str]:
    """Return a list of error strings (empty = valid). `alert` is a dict with id, rule, tray_id, ts."""
    errors: list[str] = []
    schema_errs = sorted(Draft202012Validator(assessment_schema()).iter_errors(assessment), key=lambda e: [str(p) for p in e.path])
    for e in schema_errs:
        loc = "/".join(str(p) for p in e.path) or "(root)"
        errors.append(f"schema: {loc}: {e.message[:200]}")
    if not isinstance(assessment, dict):
        return errors
    find = assessment.get("findings") if isinstance(assessment.get("findings"), list) else []
    causes = assessment.get("probable_causes") if isinstance(assessment.get("probable_causes"), list) else []
    acts = assessment.get("recommended_actions") if isinstance(assessment.get("recommended_actions"), list) else []

    # 1. citations: every evidence id must have been returned by a tool in this run, or be the alert id
    allowed = logged_ids(conn, assessment_id) | {alert["id"]}
    for name, rows in (("findings", find), ("probable_causes", causes), ("recommended_actions", acts)):
        for i, row in enumerate(rows):
            ids = row.get("evidence_ids") if isinstance(row, dict) else None
            for eid in ids if isinstance(ids, list) else []:
                if not isinstance(eid, str):
                    errors.append(f"citation: {name}[{i}] has a non-string evidence id {eid!r}")
                elif eid not in allowed:
                    errors.append(
                        f"citation: {name}[{i}] cites {eid!r}, which no tool returned in this investigation "
                        "and is not the alert id"
                    )

    # 2. likelihoods
    nums = [
        (f"{n}[{i}].{k}", row[k])
        for n, rows, k in (("probable_causes", causes, "likelihood"), ("findings", find, "confidence"))
        for i, row in enumerate(rows)
        if isinstance(row, dict) and isinstance(row.get(k), (int, float))
    ]
    errors += [f"{where} is not a finite number ({v})" for where, v in nums if not math.isfinite(v)]
    total = sum(v for where, v in nums if where.endswith("likelihood") and math.isfinite(v))
    if total > 1.0 + 1e-6:
        errors.append(f"likelihoods sum to {total:.3f}, which is more than 1.0")

    # 3. sensor_fault must be considered when an instrument fault is visible (or is the alert)
    cc = assessment.get("causes_considered")
    considered = {norm(c) for c in (cc if isinstance(cc, list) else []) if isinstance(c, str)}
    when = _as_of(conn, assessment_id, alert, as_of)
    fault_alerts = [
        a for a in visible_alerts(conn, when)
        if a["rule"] == "SENSOR_FAULT" and a["status"] == "open" and a["tray_id"] in (alert["tray_id"], "air")
    ]
    if (alert["rule"] == "SENSOR_FAULT" or fault_alerts) and "sensor_fault" not in considered:
        which = alert["id"] if alert["rule"] == "SENSOR_FAULT" else fault_alerts[0]["id"]
        errors.append(f"causes_considered must include 'sensor_fault': SENSOR_FAULT alert {which} is active")

    # 4. every recommended action needs a propose_action row from this run (same tray, same action text)
    proposals = [
        json.loads(r["proposal_json"] or "{}")
        for r in conn.execute("SELECT proposal_json FROM actions WHERE assessment_id = ?", (assessment_id,))
    ]
    have = {(p.get("tray_id"), norm(p.get("action", ""))) for p in proposals}
    tray = assessment.get("tray_id")
    for i, a in enumerate(acts):
        if isinstance(a, dict) and (not isinstance(tray, str) or (tray, norm(a.get("action", ""))) not in have):
            errors.append(
                f"action: recommended_actions[{i}] {a.get('action')!r} has no matching propose_action call for "
                f"tray {tray!r} in this run"
            )

    # 5. the assessment is about the alerted tray
    if isinstance(tray, str) and tray != alert["tray_id"]:
        errors.append(f"tray_id {tray!r} differs from the alerted tray {alert['tray_id']!r}")
    return errors
