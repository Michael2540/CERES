"""SQLite schema (PLAN.md 5.5 plus the 0.4b spend ledger and 0.6 frames.source)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS trays(
  id TEXT PRIMARY KEY, crop TEXT, sown_at TEXT, treatment TEXT,
  fault_start TEXT NULL, reference_tray_id TEXT NULL
);
CREATE TABLE IF NOT EXISTS frames(
  id TEXT PRIMARY KEY, tray_id TEXT, ts TEXT, path TEXT, cm2_per_px REAL, calib_flag TEXT,
  source TEXT,  -- dataset | fault_transform | fault_timewarp
  fault_hash TEXT NULL  -- hash of (fault params, fault_start, FAULTS_VERSION) for derived frames
);
CREATE TABLE IF NOT EXISTS plants(
  id TEXT PRIMARY KEY, tray_id TEXT, first_seen_ts TEXT, label TEXT
);
CREATE TABLE IF NOT EXISTS masks(
  id TEXT PRIMARY KEY, frame_id TEXT, tray_id TEXT, plant_id TEXT NULL, kind TEXT,
  area_px INT, area_cm2 REAL, exg REAL, chlorosis_frac REAL, rle TEXT
);
CREATE TABLE IF NOT EXISTS embeddings(
  mask_id TEXT PRIMARY KEY, model TEXT, vec BLOB, drift REAL, drift_z REAL
);
CREATE TABLE IF NOT EXISTS sensor_readings(
  id TEXT PRIMARY KEY, ts TEXT, tray_id TEXT, metric TEXT, value REAL, source TEXT
);
CREATE TABLE IF NOT EXISTS alerts(
  id TEXT PRIMARY KEY, ts TEXT, tray_id TEXT, rule TEXT, severity TEXT, evidence_ids TEXT,
  threshold_tag TEXT
);
CREATE TABLE IF NOT EXISTS assessments(
  id TEXT PRIMARY KEY, ts TEXT, tray_id TEXT, alert_id TEXT, model TEXT, provider TEXT, json TEXT,
  citation_errors TEXT, valid BOOL, latency_ms INT, cost_usd REAL, recorded BOOL
);
CREATE TABLE IF NOT EXISTS tool_log(
  assessment_id TEXT, round INT, tool TEXT, args TEXT, returned_ids TEXT
);
CREATE TABLE IF NOT EXISTS crew_notes(
  id TEXT PRIMARY KEY, ts TEXT, audio_path TEXT, transcript TEXT, extracted_json TEXT
);
CREATE TABLE IF NOT EXISTS actions(
  id TEXT PRIMARY KEY, ts TEXT, tray_id TEXT, assessment_id TEXT, proposal_json TEXT, status TEXT
);
CREATE TABLE IF NOT EXISTS link_queue(
  id TEXT PRIMARY KEY, ts TEXT, kind TEXT, payload TEXT, status TEXT, edge_json TEXT, audit_json TEXT
);
CREATE TABLE IF NOT EXISTS spend(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, role TEXT, provider TEXT, model TEXT,
  input_tokens INT, output_tokens INT, images INT, cost_usd REAL
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a connection with dict-like rows."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: str | Path) -> sqlite3.Connection:
    """Create the schema (idempotent) and return an open connection."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(frames)")}
    if "fault_hash" not in cols:  # database created before the column existed
        conn.execute("ALTER TABLE frames ADD COLUMN fault_hash TEXT NULL")
    conn.commit()
    return conn
