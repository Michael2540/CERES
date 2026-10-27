"""Tests for scripts/sensor_reader.py (simulated path)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sensor_reader import ingest_lines, parse_line, simulate_into_db, soil_vwc

from backend.config import load_trays
from backend.db import init_db
from backend.sim.sensor_hub import SimConfig

CFG = load_trays()
START = "2016-01-10T13:30:00Z"
SAMPLE = (ROOT / "tests" / "fixtures" / "serial_sample.csv").read_text().splitlines()


def one(conn, rid: str):
    return conn.execute("SELECT * FROM sensor_readings WHERE id=?", (rid,)).fetchone()


def test_parse_line() -> None:
    assert parse_line("300000,air,rh_pct,55.32") == (300000, "air", "rh_pct", 55.32)
    for bad in ("", "# c", "1,2,3", "x,air,rh_pct,1", "1,air,rh_pct,abc"):
        assert parse_line(bad) is None


def test_ids_timestamps_and_sources() -> None:
    conn = init_db(":memory:")
    ingest_lines(conn, SAMPLE, START, CFG)
    r = one(conn, "air:rh_pct:2016-01-10T13:35:00Z")
    assert (r["tray_id"], r["metric"], r["ts"], r["source"]) == (
        "air",
        "rh_pct",
        "2016-01-10T13:35:00Z",
        "simulated",
    )
    assert one(conn, "L:rh_pct:2016-01-10T13:30:00Z")["tray_id"] == "L"
    lamp = one(conn, "B:lamp_on:2016-01-10T13:30:00Z")
    assert lamp["value"] == 1.0 and lamp["source"] == "derived"
    assert one(conn, "C:ppfd_est:2016-01-10T13:30:00Z")["value"] == 80  # dimmed after fault_start
    assert one(conn, "A:ppfd_est:2016-01-10T13:30:00Z")["value"] == 200


def test_ppfd_before_low_light_fault_and_lamp_off() -> None:
    conn = init_db(":memory:")
    ingest_lines(conn, ["0,C,soil_raw,2000"], "2015-12-20T12:00:00Z", CFG)
    assert one(conn, "C:ppfd_est:2015-12-20T12:00:00Z")["value"] == 200
    ingest_lines(conn, ["0,C,soil_raw,2000"], "2015-12-20T23:00:00Z", CFG)
    assert one(conn, "C:lamp_on:2015-12-20T23:00:00Z")["value"] == 0.0
    assert one(conn, "C:ppfd_est:2015-12-20T23:00:00Z")["value"] == 0.0


def test_soil_vwc_derivation() -> None:
    assert soil_vwc(3000, 3000, 1400) == 0.0
    assert soil_vwc(1400, 3000, 1400) == 0.45
    assert soil_vwc(2200, 3000, 1400) == 0.225
    conn = init_db(":memory:")
    ingest_lines(conn, ["0,A,soil_raw,2200"], START, CFG)
    r = one(conn, "A:soil_vwc:2016-01-10T13:30:00Z")
    assert r["value"] == 0.225 and r["source"] == "derived"


def test_idempotent() -> None:
    conn = init_db(":memory:")
    n1 = ingest_lines(conn, SAMPLE, START, CFG)
    count = conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]
    assert n1 == count > 0
    assert ingest_lines(conn, SAMPLE, START, CFG) == 0
    assert conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0] == count


def test_out_of_limit_soil_raw_gets_no_derived_row() -> None:
    conn = init_db(":memory:")
    ingest_lines(conn, ["0,A,soil_raw,4095", "300000,A,soil_raw,2200"], START, CFG)
    assert one(conn, "A:soil_raw:2016-01-10T13:30:00Z")["value"] == 4095  # raw is stored as read
    assert one(conn, "A:soil_vwc:2016-01-10T13:30:00Z") is None  # but nothing is derived from it
    assert one(conn, "A:soil_vwc:2016-01-10T13:35:00Z")["value"] == 0.225


def test_resimulating_with_a_new_seed_replaces_rows() -> None:
    conn = init_db(":memory:")
    span = ("2016-01-02T00:00:00Z", "2016-01-02T03:00:00Z")
    rid = "air:temp_c:2016-01-02T01:00:00Z"
    n0 = simulate_into_db(conn, CFG, SimConfig(seed=0), *span)
    v0 = one(conn, rid)["value"]
    count = conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]
    simulate_into_db(conn, CFG, SimConfig(seed=1), *span)  # default: INSERT OR IGNORE keeps stale rows
    assert one(conn, rid)["value"] == v0
    simulate_into_db(conn, CFG, SimConfig(seed=1), *span, replace=True)
    assert one(conn, rid)["value"] != v0
    assert conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0] == count == n0
