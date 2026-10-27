"""Tests for backend/science/rules.py (pure functions, fixture series, no model calls)."""

from __future__ import annotations

import copy
import dataclasses
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sensor_reader import ingest_lines, simulate_into_db

from backend.config import load_trays
from backend.db import init_db
from backend.science import rules
from backend.science.setpoints import default_setpoints, load_setpoints
from backend.sim.sensor_hub import SimConfig, SimulatedHub, load_sim_config
from backend.timeutil import iso_z, parse_ts

CFG = load_trays()
GOLDEN = json.loads((ROOT / "tests" / "golden" / "tray_b.json").read_text())
SAMPLE = (ROOT / "tests" / "fixtures" / "serial_sample.csv").read_text().splitlines()
SAMPLE_START = "2016-01-10T13:30:00Z"


def rows(prefix: str, start: str, values: list[float], step_min: int = 5) -> list[dict]:
    t0 = parse_ts(start)
    out = []
    for i, v in enumerate(values):
        ts = iso_z(t0 + timedelta(minutes=step_min * i))
        out.append({"id": f"{prefix}:{ts}", "ts": ts, "value": v, "source": "simulated"})
    return out


def golden_inputs() -> tuple[dict, list[dict], list[dict]]:
    g = copy.deepcopy(GOLDEN)
    readings = {tuple(k.split(":")): v for k, v in g["readings"].items()}
    return readings, g["canopy"], g["ref_canopy"]


def key(a: rules.Alert) -> tuple:
    return a.rule, a.severity, a.tray_id, a.ts, list(a.evidence_ids), a.threshold_tag


def db_readings(lines: list[str], start: str) -> dict[tuple[str, str], list[dict]]:
    conn = init_db(":memory:")
    ingest_lines(conn, lines, start, CFG)
    return load_readings(conn)


def load_readings(conn) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}
    for r in conn.execute("SELECT id, ts, tray_id, metric, value, source FROM sensor_readings ORDER BY ts"):
        out.setdefault((r["tray_id"], r["metric"]), []).append(
            {"id": r["id"], "ts": r["ts"], "value": r["value"], "source": r["source"]}
        )
    return out


def with_threshold(**kw: float) -> rules.Setpoints:
    sp = default_setpoints()
    th = {k: {**v, "value": kw.get(k, v["value"])} for k, v in sp.thresholds.items()}
    return dataclasses.replace(sp, thresholds=th)


# --- setpoints -------------------------------------------------------------------------------------
def test_setpoints_have_tags_and_values() -> None:
    sp = load_setpoints()
    for name in (
        "wilt_soil_frac",
        "wilt_area_drop",
        "wilt_exg_drop",
        "wilt_window_h",
        "vpd_low",
        "low_vpd_hours",
        "flatline_min",
        "gap_min",
        "fault_lookback_min",
        "growth_lag_frac",
        "growth_window_h",
    ):
        assert sp.value(name) > 0 and sp.tag(name)
    assert sp.limit("rh_pct") == (0.0, 100.0)
    lo, hi = sp.limit("soil_raw")
    assert 0 < lo and hi < 4095  # the ADC rails are out of range
    assert {r["applies_to"] for r in sp.rows} >= {"lettuce", "arabidopsis"}
    assert all(r["tag"] == "author:control-relative" for r in sp.rows if r["applies_to"] == "arabidopsis")


# --- SENSOR_FAULT ----------------------------------------------------------------------------------
def test_sensor_fault_flatline_from_serial_sample_at_5min_cadence() -> None:
    rh = db_readings(SAMPLE, SAMPLE_START)[("air", "rh_pct")]
    a = rules.sensor_fault("air", "rh_pct", rh, "2016-01-10T14:40:00Z")
    assert a is not None and a.rule == "SENSOR_FAULT" and a.severity == "warning"
    assert a.detail.startswith("rh_pct flatline") and "2016-01-10T13:55:00Z..2016-01-10T14:40:00Z" in a.detail
    assert a.evidence_ids == ("air:rh_pct:2016-01-10T13:55:00Z", "air:rh_pct:2016-01-10T14:40:00Z")
    assert rules.sensor_fault("air", "rh_pct", rh, "2016-01-10T14:15:00Z") is None  # held < 30 min so far
    assert rules.sensor_fault("air", "rh_pct", rh, "2016-01-10T13:55:00Z") is None  # before the hold


def test_sensor_fault_latches_after_release() -> None:
    sim = load_sim_config()  # HOLD air:rh_pct 14:00, 60 min, release 15:00
    hub = SimulatedHub(CFG, sim.seed, sim.step_s, start="2016-01-10T13:30:00Z", end="2016-01-10T17:30:00Z",
                       injections=sim.injections)
    conn = init_db(":memory:")
    ingest_lines(conn, hub.lines(), "2016-01-10T13:30:00Z", CFG)
    rh = load_readings(conn)[("air", "rh_pct")]
    a = rules.sensor_fault("air", "rh_pct", rh, "2016-01-10T15:30:00Z")  # 30 min after RELEASE
    assert a is not None and "2016-01-10T13:55:00Z..2016-01-10T14:55:00Z" in a.detail
    assert rules.sensor_fault("air", "rh_pct", rh, "2016-01-10T17:00:00Z") is None  # 2 h after


def test_sensor_fault_gap_and_range() -> None:
    ok = rows("air:temp_c", "2016-01-10T10:00:00Z", [22.0 + 0.1 * (i % 3) for i in range(12)])
    assert rules.sensor_fault("air", "temp_c", ok, "2016-01-10T10:55:00Z") is None
    gap = rules.sensor_fault("air", "temp_c", ok, "2016-01-10T11:20:00Z")  # last data 10:55
    assert gap is not None and gap.detail.startswith("temp_c gap")
    holey = ok[:4] + ok[8:]
    assert rules.sensor_fault("air", "temp_c", holey, "2016-01-10T10:55:00Z").detail.startswith("temp_c gap")
    bad = ok[:-1] + rows("air:temp_c", "2016-01-10T10:55:00Z", [85.0])
    assert rules.sensor_fault("air", "temp_c", bad, "2016-01-10T10:55:00Z").detail.startswith("temp_c out of range")


# --- WILT_RISK -------------------------------------------------------------------------------------
def test_wilt_risk_positive_and_negative() -> None:
    tray = CFG.by_id("B")
    readings, canopy, _ = golden_inputs()
    soil, lamp = readings[("B", "soil_vwc")], readings[("B", "lamp_on")]
    a = rules.wilt_risk(tray, soil, canopy, lamp, "2016-01-07T18:00:00Z")
    assert a is not None and a.rule == "WILT_RISK" and a.severity == "alert" and len(a.evidence_ids) == 3
    flat = [dict(c, area_px=40000, exg=0.3) for c in canopy]  # soil dry but canopy unchanged
    assert rules.wilt_risk(tray, soil, flat, lamp, "2016-01-07T18:00:00Z") is None
    # soil still near its baseline early in the fault
    assert rules.wilt_risk(tray, soil, canopy, lamp, "2016-01-04T18:00:00Z") is None
    off = [dict(r, value=0.0) for r in lamp]  # lamp off the whole window: visual change ignored
    assert rules.wilt_risk(tray, soil, canopy, off, "2016-01-07T18:00:00Z") is None


def test_wilt_risk_exg_alone_and_no_fault_baseline() -> None:
    tray = CFG.by_id("B")
    readings, canopy, _ = golden_inputs()
    soil, lamp = readings[("B", "soil_vwc")], readings[("B", "lamp_on")]
    exg_only = [dict(c, area_px=40000) for c in canopy]
    assert rules.wilt_risk(tray, soil, exg_only, lamp, "2016-01-07T18:00:00Z") is not None
    no_fault = tray.model_copy(update={"fault_start": None})  # baseline = first 48 h of data
    w = rules.baseline_window(no_fault, soil)
    assert w is not None and w[1] - w[0] == timedelta(hours=48)
    assert rules.wilt_risk(no_fault, soil, canopy, lamp, "2016-01-07T18:00:00Z") is not None


def test_baseline_uses_only_valid_data_and_data_up_to_as_of() -> None:
    tray = CFG.by_id("A").model_copy(update={"fault_start": None})
    good = rows("A:soil_vwc", "2016-01-01T00:00:00Z", [0.35 + 0.002 * (i % 5) for i in range(576)])
    for r in good[100:500]:  # a long flatline (stuck probe) must not drag the baseline down
        r["value"] = 0.05
    for r in good[:3]:  # out-of-limit junk
        r["value"] = 9.0
    base = rules.baseline_value(tray, good, "2016-01-04T00:00:00Z")
    assert base is not None and 0.34 < base < 0.36
    # no-fault baseline only counts rows <= as_of: at 06:00 on day 1 just the first 6 h exist
    early = rules.baseline_value(tray, good, "2016-01-01T06:00:00Z")
    assert early is not None and 0.34 < early < 0.36


def test_pulled_probe_is_sensor_fault_and_suppresses_wilt() -> None:
    readings, canopy, ref = golden_inputs()
    tray, as_of = CFG.by_id("B"), GOLDEN["as_of"]
    raw, vwc = readings[("B", "soil_raw")], readings[("B", "soil_vwc")]
    for r in raw[-4:]:  # probe pulled: ADC rail value for the last 20 min
        r["value"] = 4095.0
    cut = raw[-4]["ts"]
    readings[("B", "soil_vwc")] = [r for r in vwc if r["ts"] < cut]  # no derived rows for rail values
    lamp = readings[("B", "lamp_on")]
    assert rules.wilt_risk(tray, readings[("B", "soil_vwc")], canopy, lamp, as_of) is not None  # would fire
    alerts = rules.evaluate(tray, readings, canopy, ref, as_of)
    got = {(a.rule, a.tray_id) for a in alerts}
    assert ("SENSOR_FAULT", "B") in got and ("WILT_RISK", "B") not in got
    fault = next(a for a in alerts if a.rule == "SENSOR_FAULT")
    assert "soil_raw out of range" in fault.detail


# --- LOW_VPD_RISK ----------------------------------------------------------------------------------
def vpd_rows(value: float, hours: float, start: str = "2016-01-09T09:00:00Z") -> list[dict]:
    return rows("L:vpd", start, [value] * (int(hours * 12) + 1))


def test_low_vpd_risk_positive_and_negative() -> None:
    tray = CFG.by_id("L")
    a = rules.low_vpd_risk(tray, vpd_rows(0.25, 7), [], "2016-01-09T16:00:00Z")
    assert a is not None and a.rule == "LOW_VPD_RISK" and a.severity == "watch"
    assert rules.low_vpd_risk(tray, vpd_rows(0.25, 5), [], "2016-01-09T14:00:00Z") is None  # too short
    assert rules.low_vpd_risk(tray, vpd_rows(0.9, 7), [], "2016-01-09T16:00:00Z") is None  # VPD fine
    night = vpd_rows(0.25, 7, "2016-01-09T23:00:00Z")  # lamp off 22:00-08:00
    assert rules.low_vpd_risk(tray, night, [], "2016-01-10T06:00:00Z") is None
    broken = vpd_rows(0.25, 7)
    broken[40]["value"] = 0.8  # run restarts after a normal reading
    assert rules.low_vpd_risk(tray, broken, [], "2016-01-09T16:00:00Z") is None


# --- GROWTH_LAG ------------------------------------------------------------------------------------
def canopy_rows(prefix: str, daily_growth: float, flat_from_h: float | None = None, days: int = 3) -> list[dict]:
    """Rows every 4 h from 2016-01-05T09:00Z; growth stops flat_from_h hours in (if given)."""
    t0, out = parse_ts("2016-01-05T09:00:00Z"), []
    for h in range(0, days * 24 + 1, 4):
        ts = iso_z(t0 + timedelta(hours=h))
        grown = h if flat_from_h is None else min(h, flat_from_h)
        out.append({"id": f"{prefix}:{ts}", "ts": ts, "area_px": round(10000 * (1 + daily_growth) ** (grown / 24)), "exg": 0.3})
    return out


def test_growth_lag_vs_reference() -> None:
    tray, as_of = CFG.by_id("C"), "2016-01-08T09:00:00Z"
    ref = canopy_rows("A", 0.10)
    a = rules.growth_lag(tray, canopy_rows("C", 0.02), ref, as_of)
    assert a is not None and a.rule == "GROWTH_LAG" and a.severity == "watch"
    assert rules.growth_lag(tray, canopy_rows("C", 0.09), ref, as_of) is None
    # reference tray without a reference canopy: not evaluated, no silent fallback to the own baseline
    assert rules.growth_lag(tray, canopy_rows("C", 0.02), None, as_of, own_baseline=0.10) is None


def test_growth_lag_must_hold_at_as_of_and_12h_earlier() -> None:
    tray, as_of, ref = CFG.by_id("C"), "2016-01-08T09:00:00Z", canopy_rows("A", 0.10)
    assert rules.growth_lag(tray, canopy_rows("C", 0.10, flat_from_h=42), ref, as_of) is not None  # sustained
    assert rules.growth_lag(tray, canopy_rows("C", 0.10, flat_from_h=56), ref, as_of) is None  # only the last 16 h


def test_growth_lag_own_baseline_when_no_reference() -> None:
    tray, as_of = CFG.by_id("L"), "2016-01-08T09:00:00Z"
    assert tray.reference_tray_id is None
    assert rules.growth_lag(tray, canopy_rows("L", 0.02), None, as_of, own_baseline=0.10) is not None
    assert rules.growth_lag(tray, canopy_rows("L", 0.09), None, as_of, own_baseline=0.10) is None
    assert rules.growth_lag(tray, canopy_rows("L", 0.02), None, as_of) is None  # no baseline
    c = canopy_rows("L", 0.10)
    with_fault = tray.model_copy(update={"fault_start": "2016-01-07T09:00:00Z"})
    assert rules.own_growth_baseline(with_fault, c) is not None
    assert rules.own_growth_baseline(CFG.by_id("A"), c) is None


# --- evaluate: golden, suppression, leakage ----------------------------------------------------------
def test_evaluate_golden_tray_b_exact_alert_set() -> None:
    readings, canopy, ref = golden_inputs()
    alerts = rules.evaluate(CFG.by_id("B"), readings, canopy, ref, GOLDEN["as_of"])
    got = sorted(key(a) for a in alerts)
    want = sorted(
        (e["rule"], e["severity"], e["tray_id"], e["ts"], e["evidence_ids"], e["threshold_tag"])
        for e in GOLDEN["expected_alerts"]
    )
    assert got == want
    assert {a.rule for a in alerts} == {"WILT_RISK", "GROWTH_LAG"}


def test_evaluate_skips_growth_lag_without_reference_canopy() -> None:
    readings, canopy, _ = golden_inputs()
    skipped: list[str] = []
    alerts = rules.evaluate(CFG.by_id("B"), readings, canopy, None, GOLDEN["as_of"], skipped=skipped)
    assert "GROWTH_LAG" in skipped and all(a.rule != "GROWTH_LAG" for a in alerts)


def test_evaluate_nominal_tray_a_has_no_alerts() -> None:
    readings, _, ref = golden_inputs()
    air = {k: v for k, v in readings.items() if k[0] == "air"}
    assert rules.evaluate(CFG.by_id("A"), air, ref, None, GOLDEN["as_of"]) == []


def test_soil_flatline_suppresses_wilt_risk() -> None:
    readings, canopy, ref = golden_inputs()
    raw = readings[("B", "soil_raw")]
    for r in raw[-8:]:  # held for the last 35+ min
        r["value"] = raw[-1]["value"]
    alerts = rules.evaluate(CFG.by_id("B"), readings, canopy, ref, GOLDEN["as_of"])
    got = {(a.rule, a.tray_id) for a in alerts}
    assert ("SENSOR_FAULT", "B") in got and ("WILT_RISK", "B") not in got
    assert ("GROWTH_LAG", "B") in got  # does not depend on the soil sensor


def test_rh_flatline_suppresses_low_vpd_risk_on_serial_sample() -> None:
    lines = []
    for ln in SAMPLE:  # hold tray L's dome RH 14:00-14:55, like the hub's HOLD would
        ms, tray, metric, val = ln.split(",")
        if tray == "L" and metric == "rh_pct" and 1_800_000 <= int(ms) < 5_400_000:
            val = next(x.split(",")[3] for x in SAMPLE if x.startswith("1500000,L,rh_pct"))
        lines.append(f"{ms},{tray},{metric},{val}")
    readings = db_readings(lines, SAMPLE_START)
    tray, sp = CFG.by_id("L"), with_threshold(low_vpd_hours=0.25)  # the fixture is only 2 h long
    before = rules.evaluate(tray, readings, [], None, "2016-01-10T13:55:00Z", sp=sp)
    assert [a.rule for a in before if a.tray_id == "L"] == ["LOW_VPD_RISK"]
    during = rules.evaluate(tray, readings, [], None, "2016-01-10T14:40:00Z", sp=sp)
    assert [(a.rule, a.detail[:15]) for a in during if a.tray_id == "L"] == [("SENSOR_FAULT", "rh_pct flatline")]
    assert all(a.rule != "LOW_VPD_RISK" for a in during)


def test_rh_flatline_suppresses_low_vpd_risk_synthetic() -> None:
    tray, start, as_of = CFG.by_id("L"), "2016-01-09T09:00:00Z", "2016-01-09T17:00:00Z"
    temp = rows("L:temp_c", start, [23.5 + 0.1 * (i % 4) for i in range(97)])
    rh = rows("L:rh_pct", start, [92.0 + 0.1 * (i % 3) for i in range(97)])
    readings = {("L", "temp_c"): temp, ("L", "rh_pct"): rh}
    skipped: list[str] = []
    ok = rules.evaluate(tray, readings, [], None, as_of, skipped=skipped)
    assert [a.rule for a in ok] == ["LOW_VPD_RISK"] and skipped == ["WILT_RISK", "GROWTH_LAG"]
    for r in rh[-8:]:
        r["value"] = 92.0
    held = rules.evaluate(tray, readings, [], None, as_of)
    assert [(a.rule, a.tray_id) for a in held] == [("SENSOR_FAULT", "L")]


def trunc(rows_: list[dict], as_of: str) -> list[dict]:
    return [r for r in rows_ if r["ts"] <= as_of]


@pytest.mark.parametrize("as_of", ["2016-01-05T12:00:00Z", "2016-01-06T18:00:00Z", "2016-01-07T14:00:00Z",
                                   "2016-01-07T18:00:00Z"])
def test_rules_do_not_see_the_future(as_of: str) -> None:
    readings, canopy, ref = golden_inputs()
    tray = CFG.by_id("B")
    full = rules.evaluate(tray, readings, canopy, ref, as_of)
    cut = {k: trunc(v, as_of) for k, v in readings.items()}
    assert rules.evaluate(tray, cut, trunc(canopy, as_of), trunc(ref, as_of), as_of) == full
    # poisoned future rows change nothing
    poisoned = {k: v + [dict(v[-1], id="x:" + k[1], ts="2016-01-09T00:00:00Z", value=0.0)] for k, v in readings.items()}
    assert rules.evaluate(tray, poisoned, canopy, ref, as_of) == full


# --- full-span simulated data: performance, control margin, B crossing ------------------------------
@pytest.fixture(scope="module")
def full_span():
    conn = init_db(":memory:")
    sim = load_sim_config()
    simulate_into_db(conn, CFG, SimConfig(seed=sim.seed, step_s=sim.step_s, injections=sim.injections))
    return load_readings(conn)


def test_evaluate_full_span_is_fast_per_tray(full_span) -> None:
    for tray in CFG.trays:
        t0 = time.perf_counter()
        rules.evaluate(tray, full_span, [], None, "2016-01-18T23:00:00Z")
        assert time.perf_counter() - t0 < 1.0, tray.id


def test_control_trays_keep_soil_margin_and_b_crosses_in_hours(full_span) -> None:
    sp = default_setpoints()
    for tid in ("A", "C"):
        tray = CFG.by_id(tid)
        soil = rules.Series.of(full_span[(tid, "soil_vwc")])
        base = rules.baseline_value(tray, soil, "2016-01-18T23:00:00Z")
        w = rules.baseline_window(tray, soil)
        after = [r["value"] for t, r in zip(soil.ts, soil.rows, strict=True) if t >= w[1]]
        assert min(after) / base >= 0.75, tid
    # tray A daily: no WILT soil condition even if the canopy were collapsing the whole time
    tray, soil, lamp = CFG.by_id("A"), rules.Series.of(full_span[("A", "soil_vwc")]), full_span[("A", "lamp_on")]
    t = parse_ts("2015-12-14T00:00:00Z")
    while t < parse_ts("2016-01-19T00:00:00Z"):
        t += timedelta(days=1)
        canopy = [
            {"id": f"c{k}", "ts": iso_z(t - timedelta(hours=k)), "area_px": round(100000 * 1.1**k), "exg": 0.3 * 1.1**k}
            for k in range(6, -1, -1)
        ]
        assert rules.wilt_risk(tray, soil, canopy, lamp, t) is None, iso_z(t)
    # tray B crosses 0.6 x baseline 6-12 h after fault_start
    b = CFG.by_id("B")
    soil_b = rules.Series.of(full_span[("B", "soil_vwc")])
    base_b = rules.baseline_value(b, soil_b, "2016-01-18T23:00:00Z")
    fs = parse_ts(b.fault_start)
    cross = next(t for t, r in zip(soil_b.ts, soil_b.rows, strict=True) if t >= fs and r["value"] < sp.value("wilt_soil_frac") * base_b)
    assert timedelta(hours=6) <= cross - fs <= timedelta(hours=12)
