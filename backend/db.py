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
  fault_hash TEXT NULL,  -- hash of (fault params, fault_start, FAULTS_VERSION) for derived frames
  src_file TEXT NULL  -- tray C time-warp: file name of the real source frame (data/raw/aled or aled_src)
);
CREATE TABLE IF NOT EXISTS plants(
  id TEXT PRIMARY KEY, tray_id TEXT, first_seen_ts TEXT, label TEXT
);
CREATE TABLE IF NOT EXISTS masks(
  id TEXT PRIMARY KEY, frame_id TEXT, tray_id TEXT, plant_id TEXT NULL, kind TEXT,
  area_px INT, area_cm2 REAL, exg REAL, chlorosis_frac REAL, rle TEXT,
  pots_json TEXT NULL  -- canopy rows: {pot id: area_px} per pot (backend/perception/pots.py)
);
CREATE TABLE IF NOT EXISTS embeddings(
  mask_id TEXT PRIMARY KEY, model TEXT, vec BLOB, drift REAL, drift_z REAL
);
CREATE TABLE IF NOT EXISTS sensor_readings(
  id TEXT PRIMARY KEY, ts TEXT, tray_id TEXT, metric TEXT, value REAL, source TEXT
);
CREATE TABLE IF NOT EXISTS alerts(
  id TEXT PRIMARY KEY, ts TEXT, tray_id TEXT, rule TEXT, severity TEXT, evidence_ids TEXT,
  threshold_tag TEXT,
  detail TEXT NULL,     -- latest rule detail (e.g. which sensor and the flatline span)
  status TEXT NULL,     -- open | resolved, end state of the last cycle (the API derives status per as_of)
  first_ts TEXT NULL,   -- an alert row is an episode: first cycle in which the rule fired
  last_ts TEXT NULL     -- latest cycle in which it fired
);
CREATE TABLE IF NOT EXISTS alert_events(
  alert_id TEXT, ts TEXT, evidence_ids TEXT, detail TEXT,  -- one row per cycle in which the rule fired
  PRIMARY KEY(alert_id, ts)
);
CREATE TABLE IF NOT EXISTS assessments(
  id TEXT PRIMARY KEY, ts TEXT, tray_id TEXT, alert_id TEXT, model TEXT, provider TEXT, json TEXT,
  citation_errors TEXT, valid BOOL, latency_ms INT, cost_usd REAL, recorded BOOL,
  updated_at REAL NULL,  -- heartbeat (unix time) of a running assessment, for the SSE stall check
  role TEXT NULL,       -- agronomist (NULL on old rows) | edge (Mars-mode triage, backend/link/edge.py)
  warnings TEXT NULL    -- JSON list of ungrounded numbers (backend/agent/grounding.py)
);
CREATE TABLE IF NOT EXISTS tool_log(
  assessment_id TEXT, round INT, tool TEXT, args TEXT, returned_ids TEXT,
  result TEXT NULL  -- JSON result the model saw (numeric grounding and the edge recordings read it)
);
CREATE TABLE IF NOT EXISTS crew_notes(
  id TEXT PRIMARY KEY, ts TEXT, audio_path TEXT, transcript TEXT, extracted_json TEXT
);
CREATE TABLE IF NOT EXISTS actions(
  id TEXT PRIMARY KEY, ts TEXT, tray_id TEXT, assessment_id TEXT, proposal_json TEXT, status TEXT
);
CREATE TABLE IF NOT EXISTS link_queue(
  id TEXT PRIMARY KEY, ts TEXT, kind TEXT, payload TEXT, status TEXT, edge_json TEXT, audit_json TEXT,
  queued_at REAL NULL,            -- wall-clock unix time the item was queued (delay_sim.deliverable)
  edge_assessment_id TEXT NULL,   -- assessments row (role=edge) of the on-board triage
  earth_assessment_id TEXT NULL,  -- assessments row (role=agronomist) of the Earth-side run
  error TEXT NULL
);
CREATE TABLE IF NOT EXISTS link_events(  -- every link state change; the last row is the current state
  id INTEGER PRIMARY KEY AUTOINCREMENT, state TEXT, delay_s REAL, since TEXT, since_wall REAL
);
CREATE TABLE IF NOT EXISTS spend(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, role TEXT, provider TEXT, model TEXT,
  input_tokens INT, output_tokens INT, images INT, cost_usd REAL
);
CREATE INDEX IF NOT EXISTS idx_readings ON sensor_readings(tray_id, metric, ts);
CREATE INDEX IF NOT EXISTS idx_frames_tray_ts ON frames(tray_id, ts);
CREATE INDEX IF NOT EXISTS idx_masks_frame ON masks(frame_id, kind);
CREATE INDEX IF NOT EXISTS idx_alerts_rule ON alerts(rule, tray_id, ts);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a connection with dict-like rows."""
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
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
    if "src_file" not in cols:
        conn.execute("ALTER TABLE frames ADD COLUMN src_file TEXT NULL")
    acols = {r[1] for r in conn.execute("PRAGMA table_info(alerts)")}
    for col in ("detail", "status", "first_ts", "last_ts"):
        if col not in acols:  # database created before the column existed
            conn.execute(f"ALTER TABLE alerts ADD COLUMN {col} TEXT NULL")
    if "pots_json" not in {r[1] for r in conn.execute("PRAGMA table_info(masks)")}:
        conn.execute("ALTER TABLE masks ADD COLUMN pots_json TEXT NULL")
    if "updated_at" not in {r[1] for r in conn.execute("PRAGMA table_info(assessments)")}:
        conn.execute("ALTER TABLE assessments ADD COLUMN updated_at REAL NULL")
    for table, cols in (
        ("assessments", {"role": "TEXT", "warnings": "TEXT"}),
        ("tool_log", {"result": "TEXT"}),
        ("link_queue", {"queued_at": "REAL", "edge_assessment_id": "TEXT", "earth_assessment_id": "TEXT", "error": "TEXT"}),
    ):
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for col, typ in cols.items():
            if col not in have:  # database created before the column existed
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ} NULL")
    conn.commit()
    return conn
