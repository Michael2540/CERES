"""Pipeline: run_cycle(as_of) seeds trays, ingests, segments, builds metric series, evaluates rules, raises alerts.

Never reads data after `as_of`: every query is bounded by ts <= as_of, and segmentation only touches frames
with ts <= as_of, so replaying hourly gives the same alerts as running live.

Evidence ids (documented in API.md): canopy metric points are `<tray>:<metric>:<frame ts>` (e.g.
`B:area_cm2:2016-01-03T10:00:00Z`), the same id GET /trays/{id}/series gives them and GET /evidence/{id}
resolves; readings are `<tray>:<metric>:<ts>`.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from backend.config import ROOT, TrayConfig, TraysConfig
from backend.perception import pots
from backend.perception.ingest import DERIVED_DIR, ingest_folder
from backend.perception.sam_client import SamBackend
from backend.perception.segment import segment_frames
from backend.science import rules
from backend.science.setpoints import Setpoints, default_setpoints
from backend.timeutil import iso_z, parse_ts

RAW_DIR = ROOT / "data" / "raw" / "aled"
SOWN_AT = "2015-12-14T00:00:00Z"
CLIMATE = ("temp_c", "rh_pct")


@dataclass
class CycleResult:
    as_of: str
    ingested: int = 0
    segmentation: dict | None = None
    new_alerts: list[str] = field(default_factory=list)
    updated_alerts: list[str] = field(default_factory=list)
    resolved: int = 0
    skipped: dict[str, list[str]] = field(default_factory=dict)  # tray -> rules lacking inputs


def seed_trays(conn: sqlite3.Connection, cfg: TraysConfig) -> None:
    for t in cfg.trays:
        conn.execute(
            "INSERT OR REPLACE INTO trays(id, crop, sown_at, treatment, fault_start, reference_tray_id)"
            " VALUES (?,?,?,?,?,?)",
            (t.id, t.crop, SOWN_AT, t.treatment, t.fault_start, t.reference_tray_id),
        )
    conn.commit()


def canopy_rows(conn: sqlite3.Connection, tray_id: str, as_of: str, since: str | None = None) -> list[dict]:
    """Canopy metric rows (one per segmented frame) with since <= ts <= as_of, oldest first."""
    rows = conn.execute(
        "SELECT f.id AS frame_id, f.ts, m.area_px, m.area_cm2, m.exg, m.chlorosis_frac,"
        " m.pots_json FROM masks m"
        " JOIN frames f ON f.id = m.frame_id WHERE m.kind = 'canopy' AND f.tray_id = ? AND f.ts <= ?"
        " AND f.ts >= ? ORDER BY f.ts",
        (tray_id, as_of, since or ""),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["pots"] = pots.load_pots(d.pop("pots_json"))  # pot -> area_px; None for rows without per-pot data
        out.append({**d, "id": f"{tray_id}:area_cm2:{r['ts']}"})
    return out


def reading_rows(
    conn: sqlite3.Connection, owner: str, metric: str, ranges: list[tuple[str, str]]
) -> list[dict]:
    """sensor_readings rows for owner/metric inside the (lo, hi) ranges (inclusive), oldest first."""
    cond = " OR ".join("(ts >= ? AND ts <= ?)" for _ in ranges)
    args = [x for r in ranges for x in r]
    rows = conn.execute(
        f"SELECT id, ts, value, source FROM sensor_readings WHERE tray_id = ? AND metric = ? AND ({cond})"
        " ORDER BY ts",
        (owner, metric, *args),
    ).fetchall()
    return [dict(r) for r in rows]


def _soil_ranges(
    conn: sqlite3.Connection, tray: TrayConfig, trail: str, as_of: str
) -> list[tuple[str, str]]:
    """Trailing window plus the soil baseline window (48 h before fault_start, or the first 48 h of data)."""
    out = [(trail, as_of)]
    if tray.fault_start:
        end = parse_ts(tray.fault_start)
        base = (iso_z(end - timedelta(hours=48)), iso_z(end))
    else:
        first = conn.execute(
            "SELECT MIN(ts) FROM sensor_readings WHERE tray_id = ? AND metric = 'soil_vwc' AND ts <= ?",
            (tray.id, as_of),
        ).fetchone()[0]
        if first is None:
            return out
        base = (first, iso_z(parse_ts(first) + timedelta(hours=48)))
    return [*out, (base[0], min(base[1], as_of))]


def _evaluate_tray(
    conn: sqlite3.Connection, tray: TrayConfig, as_of: datetime, cache: dict, sp: Setpoints
) -> tuple[list[rules.Alert], list[str]]:
    a_iso = iso_z(as_of)
    trail = iso_z(as_of - timedelta(hours=rules.EVAL_WINDOW_H))

    def fetch(owner: str, metric: str, ranges: list[tuple[str, str]]) -> list[dict]:
        k = (owner, metric, tuple(ranges))
        if k not in cache:
            cache[k] = reading_rows(conn, owner, metric, ranges)
        return cache[k]

    readings: dict[tuple[str, str], list[dict]] = {}
    for owner, metric in [(tray.id, "soil_raw"), (tray.id, "lamp_on")]:
        if rows := fetch(owner, metric, [(trail, a_iso)]):
            readings[(owner, metric)] = rows
    own = {m: fetch(tray.id, m, [(trail, a_iso)]) for m in CLIMATE}
    if own["temp_c"] and own["rh_pct"]:  # a tray with its own climate sensor (tray L under its dome)
        readings.update({(tray.id, m): rows for m, rows in own.items()})
    if rows := fetch(tray.id, "soil_vwc", _soil_ranges(conn, tray, trail, a_iso)):
        readings[(tray.id, "soil_vwc")] = rows
    for metric in (*CLIMATE, "co2_ppm"):
        if rows := fetch("air", metric, [(trail, a_iso)]):
            readings[("air", metric)] = rows
    if ("canopy", tray.id) not in cache:
        cache[("canopy", tray.id)] = canopy_rows(conn, tray.id, a_iso)
    canopy, ref = cache[("canopy", tray.id)], None
    if tray.reference_tray_id:
        if ("canopy", tray.reference_tray_id) not in cache:
            cache[("canopy", tray.reference_tray_id)] = canopy_rows(conn, tray.reference_tray_id, a_iso)
        ref = cache[("canopy", tray.reference_tray_id)]
    skipped: list[str] = []
    return rules.evaluate(tray, readings, canopy, ref, as_of, sp, skipped), skipped


def _fault_key(detail: str) -> str:
    """A sensor fault is identified by its metric and type (flatline|gap|range), not its span: a persistent
    fault stays one episode while it keeps firing; the episode gap separates distinct occurrences."""
    parts = detail.split()
    if len(parts) >= 3 and parts[1] == "out":
        return f"{parts[0]} range"
    return " ".join(parts[:2])


def _alert_id(conn: sqlite3.Connection, a: rules.Alert) -> str:
    base = f"alert_{a.tray_id}_{a.rule}_{a.ts.replace('-', '').replace(':', '')}"
    aid, n = base, 1
    while conn.execute("SELECT 1 FROM alerts WHERE id = ?", (aid,)).fetchone():
        n += 1
        aid = f"{base}_{n}"
    return aid


def write_alert(conn: sqlite3.Connection, a: rules.Alert) -> tuple[str, bool]:
    """Record that the rule fired at a.ts: append an event to the open episode (same rule and tray, and for
    sensor faults the same fault, last fired within rules.episode_gap_h(rule) before this cycle) or start a new episode.
    Returns (episode id, started)."""
    lo = iso_z(parse_ts(a.ts) - timedelta(hours=rules.episode_gap_h(a.rule)))
    cands = conn.execute(
        "SELECT id, detail FROM alerts WHERE rule = ? AND tray_id = ? AND last_ts >= ? AND first_ts <= ?"
        " ORDER BY last_ts DESC",
        (a.rule, a.tray_id, lo, a.ts),
    ).fetchall()
    if a.rule == "SENSOR_FAULT":
        cands = [r for r in cands if _fault_key(r["detail"] or "") == _fault_key(a.detail)]
    ev = json.dumps(list(a.evidence_ids))
    if cands:
        aid = cands[0]["id"]
        conn.execute(
            "UPDATE alerts SET last_ts = MAX(last_ts, ?), status = 'open', evidence_ids = ?, detail = ?"
            " WHERE id = ? AND last_ts <= ?",
            (a.ts, ev, a.detail, aid, a.ts),
        )
    else:
        aid = _alert_id(conn, a)
        conn.execute(
            "INSERT INTO alerts(id, ts, tray_id, rule, severity, evidence_ids, threshold_tag, detail, status,"
            " first_ts, last_ts) VALUES (?,?,?,?,?,?,?,?,'open',?,?)",
            (aid, a.ts, a.tray_id, a.rule, a.severity, ev, a.threshold_tag, a.detail, a.ts, a.ts),
        )
    conn.execute(
        "INSERT OR REPLACE INTO alert_events(alert_id, ts, evidence_ids, detail) VALUES (?,?,?,?)",
        (aid, a.ts, ev, a.detail),
    )
    return aid, not cands


def run_cycle(
    conn: sqlite3.Connection,
    cfg: TraysConfig,
    as_of: datetime | str,
    backend: SamBackend | None = None,
    segment: bool = True,
    ingest: bool = True,
    raw_dir: Path = RAW_DIR,
    derived_dir: Path = DERIVED_DIR,
    sp: Setpoints | None = None,
    cache_dir: Path | None = None,
) -> CycleResult:
    as_of = parse_ts(as_of)
    a_iso = iso_z(as_of)
    sp = sp or default_setpoints()
    res = CycleResult(as_of=a_iso)
    seed_trays(conn, cfg)
    if ingest and Path(raw_dir).is_dir():
        res.ingested = ingest_folder(conn, raw_dir, cfg, derived_dir)
    if segment and backend is not None:
        kw = {"cache_dir": cache_dir} if cache_dir else {}
        res.segmentation = segment_frames(conn, cfg, backend, as_of=a_iso, **kw)

    cache: dict = {}
    seen: set[tuple[str, str, str]] = set()  # shared-sensor faults arrive once per tray
    for tray in cfg.trays:
        alerts, skipped = _evaluate_tray(conn, tray, as_of, cache, sp)
        if skipped:
            res.skipped[tray.id] = skipped
        for a in alerts:
            k = (a.rule, a.tray_id, _fault_key(a.detail) if a.rule == "SENSOR_FAULT" else tray.id)
            if k in seen:
                continue
            seen.add(k)
            aid, inserted = write_alert(conn, a)
            (res.new_alerts if inserted else res.updated_alerts).append(aid)
    res.resolved = 0
    for rule in {r["rule"] for r in conn.execute("SELECT DISTINCT rule FROM alerts WHERE status = 'open'")}:
        cur = conn.execute(
            "UPDATE alerts SET status = 'resolved' WHERE status = 'open' AND rule = ? AND last_ts < ?",
            (rule, iso_z(as_of - timedelta(hours=rules.episode_gap_h(rule)))),
        )
        res.resolved += cur.rowcount
    conn.commit()
    return res
