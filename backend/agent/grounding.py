"""Numeric grounding: every number the model writes must come from THIS run's evidence (M8 hallucination check).

Citations prove an id was returned by a tool; they say nothing about the digits in the prose. This module extracts the
numbers of the free-text fields and checks each against the values the run actually saw. Pure functions plus one DB
reader; no model call.

Fields checked
  edge:        crew_instruction, handoff_note_for_earth
  agronomist:  findings[].claim, probable_causes[].discriminating_check, recommended_actions[].rationale / .action /
               .magnitude, setpoint_caveats[], missing_evidence[], causes_considered[], crew_brief

What counts as evidence (the Pool)
  values     every number in the RESULT of this run's tool_log rows (including numbers inside result strings), the numbers
             in the alert row's detail, and the values of the thresholds of the alert's rule. Tool ARGUMENTS are NOT
             evidence (a model can call compute_vpd(rh_pct=99.9) and then cite 99.9): compute_vpd's result echoes its
             inputs, so only its computed `vpd_kpa` enters the pool. A result that is an error is skipped.
  durations  (hours) the differences between as_of and every scalar timestamp field (`ts`, `first_ts`, ...) of the tool
             results (NOT the sample grid arrays of a series: those would ground any whole hour), the alert's first_ts ->
             as_of age, the time-unit numbers written inside the alert detail, and the look-back `hours` of each call
             (the value the tool echoed, else the argument if it is within the 72 h tool window).
  ids        the ids this run's tools returned, plus the alert id.
Rows without a stored result (runs made before the column existed) give no pool: the check is reported as unavailable
(None), never as passed.

Exact rules
  R1 masked (never checked), and ONLY these: ISO timestamps, `YYYY-MM-DD` dates (with `/DD` shorthand), compact
     `YYYYmmddTHHMMSSZ` stamps, clock times written HH:MM[:SS] with HH <= 23 and MM <= 59 ("17:83" is not a time), the exact
     id strings of this run (and the evidence-id grammar `a:b:c` only where the token is in the id set), and a fixed
     allowlist of names: rule names, cause enum values, metric names (soil_vwc, rh_pct, co2_ppm, area_cm2, ...), pot ids
     `pot\\d\\d` and "CO2"/"CO2". Everything else is read for numbers, including numbers glued to letters or punctuation:
     `RH97` is 97, `T31` is 31, `VPD:9.9` is 9.9, `soil_vwc_0.07` is 0.07, `17:83` is 17 and 83.
  R2 a bare integer <= 3 (no decimal point, no %) is a counting word ("step 1") and is ignored, unless a unit or quantity
     word follows: kPa, %, h/hours, min, days, C, ppm, umol, m3, cm2, px, units ... are checked ("VPD of 3 kPa", "dropped
     2 units", "over the last 3 days"). plants, pots, leaves, trays count only as counting words ("2 pots" is ignored; "4
     pots" is checked).
  R3 thousands commas are removed (`1,200` is 1200); the sign is not compared (`-0.4` grounds "0.4" and "-0.4"). Spelled
     numbers (zero to ninety-nine, "hundred", compounds like "thirty-eight") are converted and follow the same rules;
     "percent", "per cent" and "pct" after a number are read like `%`.
  R4 a number matches a pool value v when |v - x| <= half a unit of the last written digit (x written with d decimals:
     0.5 * 10^-d, so `0.38` matches 0.3812 and `5` matches 4.6), OR |v - x| <= 2 % of |v|.
  R5 a number followed by `%` also matches v * 100 ("38 %" grounds a soil_vwc of 0.38) and plain v ("38 %" grounds 38.4).
  R6 a FORWARD-LOOKING interval is a plan, not a claim, and is masked ONLY when its clause (split on `.` `;` `,`; not on
     "and", so "recheck soil and turgor in 12 h" is one plan) contains a plan verb (recheck, check, re-image, inspect, water, irrigate, watch, monitor, repeat, wait,
     review, follow up, reassess, confirm, retest, look again; present tense, "-s" or "-ing", never "-ed"). The pattern is
     `in|within|after|every|next N unit` (and `for N unit` only with watch|monitor|continue|wait in the clause). Every other
     duration is a claim and is checked: "soil dry for 91 hours", "fell within 37 hours", "every 91 hours it dried".
     Also masked everywhere: a forecast horizon "the next N hours" and a sampling resolution followed by a data word
     ("raw 1-min RH", "5 s readings"). In PRESCRIPTIVE fields (recommended_actions[].action/.magnitude,
     discriminating_check, crew_instruction) time-unit numbers are plans or efforts and are not checked; every other
     number there still is. Descriptive fields (findings, rationale, crew_brief, setpoint_caveats, missing_evidence,
     handoff) are checked in full.
  R7 a number with a time unit (h, hr, hours, min, days) is checked ONLY against the duration pool, with half a unit of the
     written precision as tolerance, or +-10 % when "~", "about", "around" or "approx" precedes it. No 2 % rule. The
     duration pool also holds duration setpoints and thresholds the run saw (a name ending in _h, _min or _days).
  R8 everything else is ungrounded: a number computed by the model, a threshold it remembered, a duration it invented.

Crew-action claims (non-numeric): a sentence that asserts a past crew action ("Crew watered B") with no modal or negation
must be backed by a crew note returned in this run (get_crew_notes with items); the edge has no such tool, so for the edge
every such claim is unbacked. "Crew should water B" is advice and passes.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime

from backend.science.rules import SEVERITY
from backend.science.setpoints import default_setpoints
from backend.timeutil import parse_ts

CONTEXT_CHARS = 30
COUNTING_MAX = 3
TOOL_WINDOW_H = 72

# ------------------------------------------------------------------ allowlisted names (R1)
_METRICS = (
    "soil_vwc", "soil_raw", "rh_pct", "co2_ppm", "vpd_kpa", "ppfd_est", "temp_c", "lamp_on", "area_px", "area_cm2",
    "plant_count", "growth_rate", "chlorosis_frac", "drift_z", "exg",
)
_CAUSES = (
    "underwatering", "overwatering", "low_light", "excess_light", "high_vpd", "low_vpd", "temperature_stress",
    "co2_anomaly", "nutrient_ph_issue", "sensor_fault", "camera_fault", "normal_variation", "other",
)
_NAMES = sorted({*_METRICS, *_CAUSES, *SEVERITY}, key=len, reverse=True)
_NAME_RX = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(map(re.escape, _NAMES)) + r")(?![A-Za-z])", re.IGNORECASE)
_POT_RX = re.compile(r"(?<![A-Za-z0-9])pot\d\d(?![A-Za-z0-9])", re.IGNORECASE)
_CO2_RX = re.compile(r"(?<![A-Za-z0-9])CO2(?![A-Za-z0-9])", re.IGNORECASE)

_ISO_RX = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?Z?")
_COMPACT_RX = re.compile(r"\d{8}T\d{6}Z")
_DATE_RX = re.compile(r"(?<![\d.])\d{4}-\d{2}-\d{2}(?:/\d{2})*")
_CLOCK_RX = re.compile(r"(?<![\d:.])(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?!\d)(?:\s?(?:UTC|Z)\b)?")
_ID_RX = re.compile(r"[A-Za-z_][\w.\-]*(?::[\w.\-]+)+")

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_HYPHENS = re.compile("[‐‑‒–—−]")
_NUM_RX = re.compile(r"(?<![\d.])(?P<sign>(?<![\w.])-)?(?P<num>\d+(?:\.\d+)?|\.\d+)")
_PCT_RX = re.compile(r"\s?(?:%|percent\b|per\s?cent\b|pct\b)", re.IGNORECASE)
_RANGE_RX = re.compile(r"\s*(?:-|to)\s*(?:\d+(?:\.\d+)?|\.\d+)", re.IGNORECASE)
_APPROX_RX = re.compile(r"(?:~|≈|\babout|\baround|\bapprox(?:imately|\.)?|\broughly|\bcirca)\s*$", re.IGNORECASE)

# unit words that follow a number: (regex, hours per unit) for time; quantity and counting words are plain sets
_TIME_UNITS = (
    (r"weeks?|wks?", 168.0), (r"days?|d", 24.0), (r"hours?|hrs?|h", 1.0), (r"minutes?|mins?|min", 1 / 60),
)
_TIME_RX = [(re.compile(r"[ \-]{0,2}(?:" + p + r")(?![A-Za-z])", re.IGNORECASE), h) for p, h in _TIME_UNITS]
_QTY_RX = re.compile(
    r"[ \-]{0,2}(?:kpa|hpa|pa|ppm|ppb|umol|µmol|μmol|mmol|mol|m3|cm2|cm|mm|px|units?|°\s?[cf]|c|degrees?|deg|lux|ml)"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)
_COUNT_RX = re.compile(r"[ \-]{0,2}(?:plants?|pots?|leaves|leaf|trays?|samples?|sensors?|frames?|steps?|lamps?)(?![A-Za-z])", re.IGNORECASE)

# ------------------------------------------------------------------ spelled numbers (R3)
_ONES = {w: i for i, w in enumerate((
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
    "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"))}
_TENS = {w: 10 * i for i, w in enumerate(
    ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"), start=2)}
_WORDS = sorted({*_ONES, *_TENS, "hundred"}, key=len, reverse=True)
_W = "(?:" + "|".join(_WORDS) + ")"
_WORD_RUN = re.compile(
    rf"(?<![\w])({_W})(?:(?:-|\s+(?:and\s+)?){_W})*(?![\w]|-[A-Za-z])", re.IGNORECASE
)
_WORD_TOK = re.compile(_W + "|and", re.IGNORECASE)


def _parse_spelled(run: str, base: int) -> list[tuple[int, int, int]]:
    """[(value, start, end)] for the numbers inside one run of number words ("one hundred and five twenty")."""
    toks = [(m.group(0).lower(), base + m.start(), base + m.end()) for m in _WORD_TOK.finditer(run)]
    out, i, n = [], 0, len(toks)
    while i < n:
        w = toks[i][0]
        if w == "and":
            i += 1
            continue
        s, val = i, 0
        if w in _ONES and 1 <= _ONES[w] <= 9 and i + 1 < n and toks[i + 1][0] == "hundred":
            val, i = _ONES[w] * 100, i + 2
        elif w == "hundred":
            val, i = 100, i + 1
        if val:
            if i + 1 < n and toks[i][0] == "and":
                i += 1
            if i < n and toks[i][0] in _TENS:
                val += _TENS[toks[i][0]]
                i += 1
                if i < n and toks[i][0] in _ONES and 1 <= _ONES[toks[i][0]] <= 9:
                    val += _ONES[toks[i][0]]
                    i += 1
            elif i < n and toks[i][0] in _ONES and _ONES[toks[i][0]] >= 1:
                val += _ONES[toks[i][0]]
                i += 1
        elif w in _TENS:
            val, i = _TENS[w], i + 1
            if i < n and toks[i][0] in _ONES and 1 <= _ONES[toks[i][0]] <= 9:
                val += _ONES[toks[i][0]]
                i += 1
        else:
            val, i = _ONES[w], i + 1
        out.append((val, toks[s][1], toks[i - 1][2]))
    return out


# ------------------------------------------------------------------ masking (R1) and plan masking (R6)
def _blank(text: str, rx: re.Pattern) -> str:
    return rx.sub(lambda m: " " * len(m.group(0)), text)


def _mask(text: str, ids: frozenset[str] | set[str] | tuple = ()) -> str:
    """Blank the ids of this run, timestamps, dates, clock times and the allowlisted names (and nothing else)."""
    for i in sorted(ids, key=len, reverse=True):
        if i and i in text:
            text = text.replace(i, " " * len(i))
    if ids:  # grammar fallback: an id token followed by sentence punctuation
        text = _ID_RX.sub(
            lambda m: " " * len(m.group(0)) if m.group(0).rstrip(".-") in ids else m.group(0), text)
    for rx in (_ISO_RX, _COMPACT_RX, _DATE_RX, _CLOCK_RX, _POT_RX, _CO2_RX, _NAME_RX):
        text = _blank(text, rx)
    return text


_CLAUSE_SPLIT = re.compile(r"(?<!\d)\.|\.(?!\d)|[;,\n]")  # not "and": "Recheck soil and turgor in 12 h" is one plan
_PLAN_VERB = re.compile(
    r"\b(?:re-?check|check|re-?image|image|inspect|water|irrigate|watch|monitor|repeat|wait|review|follow up|reassess|"
    r"confirm|re-?test|look again)(?:s|ing)?\b",
    re.IGNORECASE,
)
_FOR_VERB = re.compile(r"\b(?:watch|monitor|continue|wait)(?:s|ing)?\b", re.IGNORECASE)
_NUMTOK = rf"(?:\d+(?:\.\d+)?|{_W}(?:(?:-|\s+){_W})*)"
_FWD_RX = re.compile(
    rf"\b(?:in|within|after|every|next)\s+(?:about\s+|the\s+)?{_NUMTOK}(?:\s*(?:-|to)\s*{_NUMTOK})?\s*"
    r"(?:hours?|hrs?|h|minutes?|mins?|min|days?|d|weeks?)(?![A-Za-z])",
    re.IGNORECASE,
)
_FOR_RX = re.compile(
    rf"\bfor\s+(?:the\s+next\s+)?(?:about\s+)?{_NUMTOK}(?:\s*(?:-|to)\s*{_NUMTOK})?\s*"
    r"(?:hours?|hrs?|h|minutes?|mins?|min|days?|d|weeks?)(?![A-Za-z])",
    re.IGNORECASE,
)


_RESOLUTION_RX = re.compile(  # data granularity, not a measurement of the situation: "raw 1-min air RH"
    r"\b\d+(?:\.\d+)?[- ]?(?:s|sec|seconds?|min|minutes?)\b(?=[- ](?:raw|resolution|sampl\w*|interval|cadence|series|data|"
    r"readings?|logging|samples?)\b)",
    re.IGNORECASE,
)
_HORIZON_RX = re.compile(
    rf"\b(?:in\s+|over\s+|for\s+)?the\s+next\s+(?:about\s+)?{_NUMTOK}(?:\s*(?:-|to)\s*{_NUMTOK})?\s*"
    r"(?:hours?|hrs?|h|minutes?|mins?|min|days?|d|weeks?)(?![A-Za-z])",
    re.IGNORECASE,
)


def _mask_plans(text: str) -> str:
    """R6: blank forward-looking intervals, only inside a clause that carries a plan verb; "the next N hours" is a
    forecast horizon and is always blanked (it states no measurement)."""
    text = _HORIZON_RX.sub(lambda m: " " * len(m.group(0)), text)
    text = _RESOLUTION_RX.sub(lambda m: " " * len(m.group(0)), text)
    out, pos = list(text), 0
    for end in [m.start() for m in _CLAUSE_SPLIT.finditer(text)] + [len(text)]:
        seg = text[pos:end]
        rxs = ([_FWD_RX] if _PLAN_VERB.search(seg) else []) + ([_FOR_RX] if _FOR_VERB.search(seg) else [])
        for rx in rxs:
            for m in rx.finditer(seg):
                for k in range(pos + m.start(), pos + m.end()):
                    out[k] = " "
        pos = end
    return "".join(out)


def _prep(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    return _THOUSANDS.sub("", _HYPHENS.sub("-", t))


def _unit_after(t: str, end: int) -> tuple[str | None, float | None]:
    """('time', hours per unit) | ('qty', None) | ('count', None) | (None, None) for the word after a number at `end`."""
    rest = t[end:]
    while True:  # a range "6-12 h": the unit belongs to the whole range
        m = _RANGE_RX.match(rest)
        if not m:
            break
        rest = rest[m.end():]
    for rx, hours in _TIME_RX:
        if rx.match(rest):
            return "time", hours
    if _QTY_RX.match(rest):
        return "qty", None
    if _COUNT_RX.match(rest):
        return "count", None
    return None, None


def extract(text: str, drop_counting: bool = True, ids=(), spelled: bool | None = None) -> list[dict]:
    """Numbers of `text` after R1-R3 (and R6 for model prose; evidence keeps its durations, `drop_counting=False`):
    [{text, value (abs), decimals, pct, start, unit, unit_hours, approx}]. `ids` are this run's exact id strings."""
    t = _mask(_prep(text), ids)
    if drop_counting:
        t = _mask_plans(t)
    spelled = drop_counting if spelled is None else spelled
    raw: list[tuple[int, int, float, int, str]] = []  # (start, end of number, value, decimals, shown text)
    for m in _NUM_RX.finditer(t):
        body = m.group("num")
        raw.append((m.start(), m.end(), float(body), len(body.split(".")[1]) if "." in body else 0, m.group(0)))
    if spelled:
        for run in _WORD_RUN.finditer(t):
            for val, s, e in _parse_spelled(run.group(0), run.start()):
                raw.append((s, e, float(val), 0, t[s:e]))
    out = []
    for s, e, value, dec, shown in sorted(raw):
        pm = _PCT_RX.match(t, e)
        pct = pm is not None
        unit, hours = (None, None) if pct else _unit_after(t, e)
        if (drop_counting and dec == 0 and not pct and value <= COUNTING_MAX and unit in (None, "count")):
            continue
        out.append({
            "text": (shown + (pm.group(0) if pm else "")).strip(), "value": abs(value), "decimals": dec, "pct": pct,
            "start": s, "unit": unit, "unit_hours": hours, "approx": bool(_APPROX_RX.search(t[:s])),
        })
    return out


# ------------------------------------------------------------------ the pool
@dataclass
class Pool:
    values: list[float] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)  # hours
    ids: set[str] = field(default_factory=set)


def _as_pool(pool) -> Pool:
    return pool if isinstance(pool, Pool) else Pool(values=list(pool))


def matches(num: dict, pool) -> bool:
    """R4 and R5 (plain values), or R7 for a number with a time unit (against the duration pool)."""
    p = _as_pool(pool)
    x = num["value"]
    tol_abs = 0.5 * 10 ** -num["decimals"] + 1e-9
    if num.get("unit") == "time":
        for hours in p.durations:
            c = hours / num["unit_hours"]  # the pool value expressed in the unit that was written
            tol = max(tol_abs, 0.10 * abs(c)) if num.get("approx") else tol_abs
            if abs(c - x) <= tol:
                return True
        return False
    for v in p.values:
        for c in ((v, v * 100) if num["pct"] else (v,)):
            if abs(c - x) <= tol_abs or abs(c - x) <= 0.02 * abs(c) + 1e-12:
                return True
    return False


_DUR_SUFFIX = {"h": 1.0, "hours": 1.0, "min": 1 / 60, "days": 24.0}  # name suffix of a duration setpoint -> hours per unit


def collect(obj, out: list[float] | None = None, dur: list[float] | None = None) -> list[float]:
    """Every number inside a JSON value (leaves and numbers written inside strings), absolute values. Time-unit numbers
    written inside strings are also added to `dur` (hours) when given."""
    out = [] if out is None else out
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out.append(abs(float(obj)))
    elif isinstance(obj, str):
        for n in extract(obj, drop_counting=False):
            out.append(n["value"])
            if dur is not None and n["unit"] == "time":
                dur.append(n["value"] * n["unit_hours"])
    elif isinstance(obj, dict):
        if dur is not None:  # a duration setpoint ({"name": "wilt_window_h", "value": 24} or {"wilt_window_h": 24})
            name = next((obj[k] for k in ("name", "id") if isinstance(obj.get(k), str)), "")
            val = obj.get("value")
            for key, num in [(name, val), *obj.items()]:
                per_h = _DUR_SUFFIX.get(str(key).rsplit("_", 1)[-1]) if "_" in str(key) else None
                if per_h and isinstance(num, (int, float)) and not isinstance(num, bool) and math.isfinite(num):
                    dur.append(abs(float(num)) * per_h)
        for k, v in obj.items():
            if k != "_error":
                collect(v, out, dur)
    elif isinstance(obj, list):
        for v in obj:
            collect(v, out, dur)
    return out


_TS_KEYS = ("ts", "first_ts", "last_ts", "since", "as_of", "recorded_at")


def _scalar_timestamps(obj, out: list[str]) -> None:
    """Timestamps held in scalar `ts`-like fields (never the sample-grid arrays of a series)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _TS_KEYS and isinstance(v, str):
                out.append(v)
            elif isinstance(v, (dict, list)):
                _scalar_timestamps(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _scalar_timestamps(v, out)


def _hours_between(a: datetime, ts: str) -> float | None:
    try:
        return abs((a - parse_ts(ts)).total_seconds()) / 3600.0
    except (ValueError, TypeError):
        return None


def _as_of(conn: sqlite3.Connection, assessment_id: str, alert: dict) -> datetime | None:
    r = conn.execute("SELECT ts FROM assessments WHERE id = ?", (assessment_id,)).fetchone()
    try:
        return parse_ts(r["ts"] if r and r["ts"] else alert["ts"])
    except (KeyError, ValueError, TypeError):
        return None


def pool_from_calls(calls: list[dict], as_of: datetime | None, alert: dict, first_ts: str | None = None) -> Pool:
    """The Pool of one run from its calls [{tool, args (dict), result (parsed JSON or None), returned_ids}] (pure: the DB
    reader and the offline re-check of a recording both go through here)."""
    p = Pool(ids={alert["id"]} if alert.get("id") else set())
    for c in calls:
        res, args = c.get("result"), c.get("args") if isinstance(c.get("args"), dict) else {}
        p.ids.update(c.get("returned_ids") or [])
        if isinstance(res, dict) and "error" in res and len(res) == 1:
            continue
        if c.get("tool") == "compute_vpd" and isinstance(res, dict):
            res = {"vpd_kpa": res.get("vpd_kpa")}  # the result echoes temp_c / rh_pct, which the model chose itself
        collect(res, p.values, p.durations)
        if as_of is not None:
            stamps: list[str] = []
            _scalar_timestamps(res, stamps)
            p.durations += [h for h in (_hours_between(as_of, s) for s in stamps) if h is not None]
        hrs = res.get("hours") if isinstance(res, dict) else None
        if not isinstance(hrs, (int, float)) or isinstance(hrs, bool):
            hrs = args.get("hours") if res is not None else None
        if isinstance(hrs, (int, float)) and not isinstance(hrs, bool) and 0 < hrs <= TOOL_WINDOW_H:
            p.durations.append(float(hrs))
    collect(alert.get("detail") or "", p.values, p.durations)
    collect(alert.get("threshold_tag") or "", p.values, p.durations)
    if first_ts and as_of is not None:
        h = _hours_between(as_of, first_ts)
        if h is not None:
            p.durations.append(h)
    for t in default_setpoints().thresholds.values():
        if t.get("rule") == alert.get("rule"):
            collect(t.get("value"), p.values)
    return p


def build_pool(conn: sqlite3.Connection, assessment_id: str, alert: dict) -> Pool | None:
    """What this run saw (tool RESULTS, the alert row, its rule's thresholds); None if results were not stored."""
    rows = conn.execute(
        "SELECT tool, args, result, returned_ids FROM tool_log WHERE assessment_id = ?", (assessment_id,)).fetchall()
    if rows and all(r["result"] is None for r in rows):
        return None
    calls = []
    for r in rows:
        try:
            calls.append({
                "tool": r["tool"], "args": json.loads(r["args"] or "{}"), "returned_ids": json.loads(r["returned_ids"] or "[]"),
                "result": json.loads(r["result"]) if r["result"] else None,
            })
        except json.JSONDecodeError:
            continue
    fr = conn.execute("SELECT first_ts FROM alerts WHERE id = ?", (alert.get("id"),)).fetchone()
    return pool_from_calls(calls, _as_of(conn, assessment_id, alert), alert, fr["first_ts"] if fr else None)


# Prescriptive fields tell someone what to do; a time interval there is a plan or an effort ("5 min visual check"), never
# a measurement, so time-unit numbers are not checked in them (every other number still is). Descriptive fields
# (findings, rationale, crew_brief, setpoint_caveats, missing_evidence, handoff) are checked in full: "no data for 91
# hours" is a claim. A sampling resolution ("1-min raw RH", "5 s readings") is masked everywhere (_RESOLUTION_RX).
_PRESCRIPTIVE = re.compile(r"(?:^|\.)(?:action|magnitude|discriminating_check|crew_instruction)$")


def ungrounded(texts: dict[str, str], pool) -> list[dict]:
    """[{field, number, context}] for every number of `texts` ({field label: text}) that no pool value grounds."""
    p = _as_pool(pool)
    bad = []
    for fld, text in texts.items():
        plan_field = bool(_PRESCRIPTIVE.search(fld))
        for n in extract(text, ids=p.ids):
            if plan_field and n["unit"] == "time":
                continue
            if not matches(n, p):
                s = n["start"]
                ctx = (text or "")[max(0, s - CONTEXT_CHARS): s + len(n["text"]) + CONTEXT_CHARS].strip()
                bad.append({"field": fld, "number": n["text"], "context": ctx})
    return bad


def format_errors(bad: list[dict]) -> list[str]:
    return [
        f"grounding: number {b['number']!r} in {b['field']} ('...{b['context']}...') is not in any tool result, alert or "
        "setpoint of this run (a duration must be a time difference a tool result shows); remove it or use a number a "
        "tool returned"
        for b in bad
    ]


# ------------------------------------------------------------------ crew-action claims
_CREW = r"(?:crew|astronauts?|someone|operators?)"
_PAST_ACT = r"(?:watered|irrigated|adjusted|moved|replaced|reseated|turned|fixed|refilled|checked)"
_CLAIM_A = re.compile(rf"\b{_CREW}\b[^.;!?\n]{{0,40}}?\b{_PAST_ACT}\b", re.IGNORECASE)
_CLAIM_B = re.compile(rf"\b{_PAST_ACT}\b[^.;!?\n]{{0,30}}?\bby\s+(?:the\s+)?{_CREW}\b", re.IGNORECASE)
_MODAL = re.compile(
    r"\b(?:should|must|please|recommend\w*|can|may|could|will|not|never|if|unless|whether)\b|n't", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<!\d)\.|\.(?!\d)|[;!?\n]")


def crew_claims(text: str) -> list[str]:
    """Sentences that assert a past crew action without a modal or negation ("Crew watered B")."""
    out = []
    for s in _SENT_SPLIT.split(unicodedata.normalize("NFKC", text or "")):
        if (_CLAIM_A.search(s) or _CLAIM_B.search(s)) and not _MODAL.search(s):
            out.append(s.strip())
    return out


def crew_backed(conn: sqlite3.Connection, assessment_id: str) -> bool:
    """True when a get_crew_notes call of this run returned at least one note."""
    for r in conn.execute(
        "SELECT result FROM tool_log WHERE assessment_id = ? AND tool = 'get_crew_notes'", (assessment_id,)
    ):
        try:
            res = json.loads(r["result"] or "null")
        except json.JSONDecodeError:
            continue
        if isinstance(res, dict) and res.get("items"):
            return True
    return False


def crew_claim_hits(texts: dict[str, str], backed: bool) -> list[dict]:
    """[{field, number, context}] for each unbacked crew-action claim (number holds the label, context the sentence)."""
    if backed:
        return []
    return [
        {"field": fld, "number": "unbacked crew-action claim", "context": s[:120]}
        for fld, text in texts.items() for s in crew_claims(text)
    ]


def crew_errors(texts: dict[str, str], backed: bool) -> list[str]:
    return [f"unbacked crew-action claim in {h['field']}: '{h['context']}' (no crew note was returned in this run)"
            for h in crew_claim_hits(texts, backed)]


# ------------------------------------------------------------------ fields
def assessment_texts(a: dict) -> dict[str, str]:
    """The free-text fields of an agronomist assessment that are number-checked."""
    out: dict[str, str] = {}
    if not isinstance(a, dict):
        return out
    for key, fields in (
        ("findings", ("claim",)), ("probable_causes", ("discriminating_check",)),
        ("recommended_actions", ("rationale", "action", "magnitude")),
    ):
        rows = a.get(key)
        for i, row in enumerate(rows if isinstance(rows, list) else []):
            for fld in fields:
                if isinstance(row, dict) and isinstance(row.get(fld), str):
                    out[f"{key}[{i}].{fld}"] = row[fld]
    for key in ("setpoint_caveats", "missing_evidence", "causes_considered"):
        rows = a.get(key)
        for i, v in enumerate(rows if isinstance(rows, list) else []):
            if isinstance(v, str):
                out[f"{key}[{i}]"] = v
    if isinstance(a.get("crew_brief"), str):
        out["crew_brief"] = a["crew_brief"]
    return out


def edge_texts(e: dict) -> dict[str, str]:
    return {k: e[k] for k in ("crew_instruction", "handoff_note_for_earth") if isinstance(e, dict) and isinstance(e.get(k), str)}


def check(conn: sqlite3.Connection, assessment_id: str, alert: dict, texts: dict[str, str]) -> list[dict] | None:
    """Ungrounded numbers of `texts` for a stored run; None when the run has no stored tool results."""
    pool = build_pool(conn, assessment_id, alert)
    return None if pool is None else ungrounded(texts, pool)


def store_warnings(conn: sqlite3.Connection, assessment_id: str, assessment: dict | None, alert: dict) -> list[dict] | None:
    """Agronomist: write the ungrounded numbers and unbacked crew claims to assessments.warnings (a WARNING, never a
    validity failure)."""
    if not isinstance(assessment, dict):
        return None
    texts = assessment_texts(assessment)
    bad = check(conn, assessment_id, alert, texts)
    if bad is not None:
        bad = bad + crew_claim_hits(texts, crew_backed(conn, assessment_id))
    conn.execute("UPDATE assessments SET warnings = ? WHERE id = ?", (None if bad is None else json.dumps(bad), assessment_id))
    conn.commit()
    return bad
