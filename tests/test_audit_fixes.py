"""Regression tests for the adversarial-audit fixes: per-pot tracking (ALED destructive sampling), migrations,
WILT_RISK latching, validator hardening, fault disclosure, wilt ghosts, ROI margin, cache keys, lamp schedules.

All offline: synthetic frames, HSV backend, scripted LLM from test_agent. No Meta API calls."""

from __future__ import annotations

import itertools
import json
import re
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_agent import (  # noqa: F401  (fixtures and helpers are shared with the agent tests)
    ALERT_ROW,
    AS,
    AS_OF,
    SOIL,
    _logged,
    _propose,
    base_db,
    db,
    final_json,
)
from test_api import api  # noqa: F401
from test_pipeline import CFG, _frame

from backend.agent.tools import ToolContext, run_tool
from backend.agent.validate import validate
from backend.api import deps
from backend.app import app
from backend.config import ROOT, FaultParams, PotGrid
from backend.db import init_db
from backend.perception import metrics, pots, segment
from backend.perception.ingest import fault_hash
from backend.perception.sam_client import HsvBackend, Mask, cache_key, legacy_cache_key
from backend.perception.segment import (
    CacheMiss,
    estimate_cost_usd,
    migrate_cache,
    recompute_from_cache,
    segment_frames,
    write_masks,
)
from backend.science import rules
from backend.sim.faults import FAULTS_VERSION, plant_mask, wilt_transform
from backend.timeutil import iso_z, lamp_is_on, parse_ts

sys.path.insert(0, str(ROOT / "scripts"))
from backfill_pots import backfill

A, B, C = CFG.by_id("A"), CFG.by_id("B"), CFG.by_id("C")


# ------------------------------------------------------------------ 1. pot grid and assignment
def test_pot_grid_is_an_even_5x4_split_with_optional_overrides():
    assert all(t.pot_grid.cols == 5 and t.pot_grid.rows == 4 and t.roi_margin_px == 40 for t in CFG.trays)
    b = pots.pot_bounds(A)
    assert len(b) == 20 and b[(0, 0)] == (65, 15, 246, 200) and b[(3, 4)][2:] == (970, 755)
    assert pots.pot_of(A, 70, 20) == (0, 0) and pots.pot_of(A, 245.9, 199.9) == (0, 0)
    assert pots.pot_of(A, 246, 200) == (1, 1) and pots.pot_of(A, 960, 750) == (3, 4)
    assert pots.pot_of(A, 5, 5) == (0, 0) and pots.pot_of(A, 1020, 765) == (3, 4)  # margin: nearest cell
    odd = A.model_copy(update={"pot_grid": PotGrid(cells={"0,0": (0, 0, 300, 100)})})
    assert pots.pot_of(odd, 290, 50) == (0, 0) and pots.pot_of(A, 290, 50) == (0, 1)
    assert pots.pot_bounds(odd)[(0, 0)] == (0, 0, 300, 100)
    assert pots.pot_id("A", 2, 3) == "A:pot23"


def _rect(x0, y0, x1, y1, shape=(768, 1024)) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_assign_unions_instances_of_one_pot():
    ids, area = pots.assign(A, [_rect(100, 60, 140, 100), _rect(120, 80, 160, 120), _rect(800, 600, 840, 640), _rect(0, 0, 0, 0)])
    assert ids == ["A:pot00", "A:pot00", "A:pot34", None]
    assert area == {"A:pot00": 40 * 40 + 40 * 40 - 20 * 20, "A:pot34": 1600}  # union, not a double count


def _mask(r: np.ndarray, i: int = 0) -> Mask:
    ys, xs = np.nonzero(r)
    return Mask(str(i), (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), 0.9, metrics.rle_encode(r))


def _write(conn, tmp_path, tray_id: str, ts: str, masks: list[Mask]) -> sqlite3.Row:
    _frame(conn, tmp_path, tray_id, ts, 10)
    conn.commit()
    row = conn.execute("SELECT id, tray_id, ts, path, cm2_per_px, fault_hash FROM frames WHERE id = ?", (f"{tray_id}:{ts}",)).fetchone()
    img = cv2.imread(str(ROOT / row["path"]))
    for m in masks:  # the colour trim keeps only plant-coloured pixels, so the masks must sit on green
        img[m.raster()] = (30, 170, 60)
    cv2.imwrite(str(ROOT / row["path"]), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    write_masks(conn, row, masks, CFG.by_id(tray_id))
    return row


def test_write_masks_assigns_pots_and_clips_to_roi_plus_margin(tmp_path):
    conn = init_db(":memory:")
    rects = [
        _rect(130, 80, 170, 120),  # pot00, 1600 px
        _rect(800, 600, 840, 640),  # pot34, 1600 px
        _rect(300, 250, 305, 255),  # pot11, 25 px: a speck, not a plant
        _rect(1000, 300, 1020, 340),  # beyond the ROI (x < 970) but inside the 40 px margin up to 1010: 10 x 40
        _rect(1015, 500, 1023, 540),  # beyond ROI + margin: dropped
    ]
    row = _write(conn, tmp_path, "A", "2016-01-06T10:00:00Z", [_mask(r, i) for i, r in enumerate(rects)])
    canopy = conn.execute("SELECT * FROM masks WHERE kind = 'canopy'").fetchone()
    assert canopy["area_px"] == 1600 + 1600 + 25 + 400
    assert json.loads(canopy["pots_json"]) == {"A:pot00": 1600, "A:pot11": 25, "A:pot14": 400, "A:pot34": 1600}
    plant_rows = {r["id"]: r["plant_id"] for r in conn.execute("SELECT id, plant_id FROM masks WHERE kind = 'plant'")}
    assert plant_rows == {f"{row['id']}:plant:0": "A:pot00", f"{row['id']}:plant:1": "A:pot34",
                          f"{row['id']}:plant:2": None, f"{row['id']}:plant:3": "A:pot14"}  # the speck holds no plant
    plants = {r["id"]: r["first_seen_ts"] for r in conn.execute("SELECT * FROM plants")}
    assert plants == {"A:pot00": row["ts"], "A:pot14": row["ts"], "A:pot34": row["ts"]}
    # an earlier frame moves first_seen_ts back; a later one does not
    _write(conn, tmp_path, "A", "2016-01-05T10:00:00Z", [_mask(rects[0])])
    _write(conn, tmp_path, "A", "2016-01-07T10:00:00Z", [_mask(rects[0])])
    assert conn.execute("SELECT first_seen_ts FROM plants WHERE id = 'A:pot00'").fetchone()[0] == "2016-01-05T10:00:00Z"


def test_backfill_reproduces_write_masks_and_is_idempotent(tmp_path):
    conn = init_db(":memory:")
    rects = [_rect(130, 80, 170, 120), _rect(800, 600, 840, 640)]
    row = _write(conn, tmp_path, "A", "2016-01-06T10:00:00Z", [_mask(r, i) for i, r in enumerate(rects)])
    want = conn.execute("SELECT id, plant_id, pots_json FROM masks ORDER BY id").fetchall()
    want = [tuple(r) for r in want]
    conn.execute("UPDATE masks SET plant_id = NULL, pots_json = NULL")
    conn.execute("DELETE FROM plants")
    for _ in range(2):  # idempotent
        stats = backfill(conn, CFG)
        assert stats["frames"] == 1 and stats["plants"] == 2
        assert [tuple(r) for r in conn.execute("SELECT id, plant_id, pots_json FROM masks ORDER BY id")] == want
    assert conn.execute("SELECT first_seen_ts FROM plants WHERE id = 'A:pot00'").fetchone()[0] == row["ts"]


# ------------------------------------------------------------------ 1. growth on tracked pots
def crow(prefix: str, h: int, per_pot: dict[str, int], t0: str = "2016-01-05T09:00:00Z") -> dict:
    ts = iso_z(parse_ts(t0) + timedelta(hours=h))
    return {"id": f"{prefix}:area_cm2:{ts}", "ts": ts, "area_px": sum(per_pot.values()), "exg": 0.3, "pots": dict(per_pot)}


def growing(prefix: str, remove_at_h: int | None = None, days: int = 4, rate: float = 0.10, strip: bool = False) -> list[dict]:
    """Five pots growing `rate` per day; the last pot is emptied from remove_at_h on (destructive sampling)."""
    out = []
    for h in range(0, days * 24 + 1, 4):
        pp = {f"{prefix}:pot0{k}": round(2000 * (1 + rate) ** (h / 24)) for k in range(5)}
        if remove_at_h is not None and h >= remove_at_h:
            pp[f"{prefix}:pot04"] = 0
        row = crow(prefix, h, pp)
        if strip:
            row.pop("pots")
        out.append(row)
    return out


def test_growth_rate_24h_ignores_pots_emptied_inside_the_window():
    rows = growing("A", remove_at_h=84)  # the window (72, 96] straddles the removal
    ts, area, per_pot = [parse_ts(r["ts"]) for r in rows], [r["area_px"] for r in rows], [r["pots"] for r in rows]
    at = ts[-1]
    assert metrics.growth_rate_24h(ts, area, at) < 0  # the total sees a 20 % drop
    assert metrics.growth_rate_24h(ts, area, at, per_pot) == pytest.approx(np.log(1.10), rel=0.05)
    assert metrics.growth_rate_24h(ts, area, at, [None] * len(ts)) is None  # no per-pot data: nothing to track


def test_growth_lag_does_not_fire_on_a_removal_but_still_on_real_lag():
    ref = growing("A")
    own = growing("C", remove_at_h=70)
    as_of = ref[21]["ts"]  # 84 h: both 24 h windows (and the one 12 h earlier) straddle the removal at 70 h
    assert rules.growth_lag(C, own, ref, as_of) is None  # a pot was emptied; the rest grows like the reference
    stripped = growing("C", remove_at_h=70, strip=True)
    assert rules.growth_lag(C, stripped, [dict(r, pots=None) for r in ref], as_of) is not None  # total: false lag
    slow = growing("C", rate=0.02, remove_at_h=70)
    assert rules.growth_lag(C, slow, ref, as_of) is not None


def test_reference_tray_uses_the_same_window_logic():
    own, ref = growing("C"), growing("A", remove_at_h=70)  # the REFERENCE loses a pot
    assert rules.growth_lag(C, own, ref, own[21]["ts"]) is None
    g = rules._growth(rules.Series.of(ref), parse_ts(ref[21]["ts"]), 24)
    assert g is not None and g[0] == pytest.approx(0.10, rel=0.05)


# ------------------------------------------------------------------ 1/3. WILT_RISK on tracked pots, latching
SOIL_LOW = 0.05
FAULT = parse_ts(B.fault_start)


def wilt_canopy(days: int = 12, remove_at_d: float | None = None, recover_from_d: float | None = None) -> list[dict]:
    """Rows every 4 h from 7 days before fault_start: five pots at 2000 px, shrinking by s(d)^2 after it."""
    out = []
    for h in range(-7 * 24, days * 24 + 1, 4):
        d = max(h / 24, 0.0)
        s = max(0.55, 1 - 0.05 * d)
        if recover_from_d is not None and d >= recover_from_d:
            s = 1.0
        pp = {f"B:pot0{k}": round(2000 * s * s) for k in range(5)}
        if remove_at_d is not None and d >= remove_at_d:
            pp["B:pot04"] = 0
        ts = iso_z(FAULT + timedelta(hours=h))
        out.append({"id": f"B:area_cm2:{ts}", "ts": ts, "area_px": sum(pp.values()), "exg": 0.3, "pots": pp})
    return out


def soil_rows(value: float, as_of: datetime) -> list[dict]:
    ts = iso_z(as_of)
    return [{"id": f"B:soil_vwc:{ts}", "ts": ts, "value": value, "source": "simulated"}]


def wilt(canopy, as_of: datetime, soil: float = SOIL_LOW, latch: bool = True):
    hist = [r for r in canopy if parse_ts(r["ts"]) <= as_of]
    win = [r for r in hist if parse_ts(r["ts"]) >= as_of - timedelta(hours=72)]
    return rules.wilt_risk(B, soil_rows(soil, as_of), win, [], as_of, base_value=0.30, history=hist if latch else None)


def test_wilt_risk_latches_through_a_whole_drought():
    cano = wilt_canopy()
    late = FAULT + timedelta(days=11, hours=3)  # shrinking stopped on day 9: no drop inside the last 24 h
    assert wilt(cano, late, latch=False) is None
    a = wilt(cano, late)
    assert a is not None and a.rule == "WILT_RISK" and "latched" in a.detail
    assert a.evidence_ids[0].startswith("B:soil_vwc:") and any(i.startswith("B:area_cm2:") for i in a.evidence_ids[1:])
    # it also fires through the lamp-off night, when the last frame is hours old
    assert wilt(cano, FAULT + timedelta(days=6, hours=18)) is not None  # 02:00 UTC: lamp off


def test_one_drought_is_one_episode():
    cano, fired = wilt_canopy(), []
    t = FAULT
    while t <= FAULT + timedelta(days=12):
        if wilt(cano, t) is not None:
            fired.append(t)
        t += timedelta(hours=1)
    assert fired and fired[0] - FAULT < timedelta(days=2)
    gaps = [b - a for a, b in itertools.pairwise(fired)]
    assert max(gaps) <= timedelta(hours=1)  # fires every cycle from the first firing to the end: one episode
    assert fired[-1] >= FAULT + timedelta(days=12) - timedelta(hours=1)


def test_wilt_latch_ends_when_soil_or_canopy_recover():
    late = FAULT + timedelta(days=11, hours=3)
    assert wilt(wilt_canopy(), late, soil=0.25) is None  # soil back above 0.6 x baseline
    assert wilt(wilt_canopy(recover_from_d=10), late) is None  # canopy back at its pre-fault peak
    assert wilt(wilt_canopy(), FAULT - timedelta(hours=2)) is None  # not before fault_start


def test_wilt_latch_ignores_lamp_off_frames_and_rows_without_pots():
    flat = wilt_canopy(days=3, recover_from_d=0)  # healthy: nothing should latch
    late = FAULT + timedelta(days=3)
    assert wilt(flat, late) is None
    dark_ts = next(r["ts"] for r in reversed(flat) if not lamp_is_on(B.lamp, parse_ts(r["ts"])))
    dim = [dict(r, exg=0.05) if r["ts"] == dark_ts else r for r in flat]  # one dark frame after the fault
    assert wilt(dim, parse_ts(dark_ts)) is None
    removed = [dict(r, pots=None, area_px=r["area_px"] * (0.6 if parse_ts(r["ts"]) >= FAULT else 1)) for r in flat]
    assert wilt(removed, late) is None  # a total-area drop alone (e.g. plants removed) is not latched


def test_wilt_risk_ignores_a_pot_emptied_by_sampling():
    cano = wilt_canopy(days=0)  # pre-fault: five pots, flat
    flat = [dict(r, pots={**r["pots"], "B:pot04": 2000}) for r in cano]
    as_of = FAULT + timedelta(hours=2)
    removal = flat + [crow("B", 0, {**{f"B:pot0{k}": 2000 for k in range(4)}, "B:pot04": 0}, t0=iso_z(FAULT))]
    removal[-1]["ts"] = iso_z(as_of)
    removal[-1]["id"] = f"B:area_cm2:{iso_z(as_of)}"
    assert wilt(removal, as_of) is None  # total area is down 20 %, the tracked pots are flat
    stripped = [dict(r, pots=None) for r in removal]
    assert wilt(stripped, as_of, latch=False) is not None  # without tracking this is a (false) wilt alarm


# ------------------------------------------------------------------ 1. API: plant_count, evidence
def test_plant_count_series_and_evidence(api):  # noqa: F811
    r = api.get("/trays/B/series", params={"metric": "plant_count"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["unit"] == "plants" and body["items"] and all(p["value"] == 1 for p in body["items"])
    pid = body["items"][-1]["id"]
    assert pid == f"B:plant_count:{body['items'][-1]['ts']}"
    ev = api.get(f"/evidence/{pid}")
    assert ev.status_code == 200 and ev.json()["metric"] == "plant_count" and ev.json()["value"] == 1


# ------------------------------------------------------------------ 2. migrations
def test_open_db_migrates_an_old_schema_db(tmp_path, monkeypatch):
    p = tmp_path / "old.db"
    old = sqlite3.connect(p)
    old.executescript(
        "CREATE TABLE assessments(id TEXT PRIMARY KEY, ts TEXT, tray_id TEXT, alert_id TEXT, model TEXT, provider TEXT,"
        " json TEXT, citation_errors TEXT, valid BOOL, latency_ms INT, cost_usd REAL, recorded BOOL);"
        "CREATE TABLE masks(id TEXT PRIMARY KEY, frame_id TEXT, tray_id TEXT, plant_id TEXT NULL, kind TEXT, area_px INT,"
        " area_cm2 REAL, exg REAL, chlorosis_frac REAL, rle TEXT);"
        "CREATE TABLE frames(id TEXT PRIMARY KEY, tray_id TEXT, ts TEXT, path TEXT, cm2_per_px REAL, calib_flag TEXT, source TEXT);"
        "CREATE TABLE alerts(id TEXT PRIMARY KEY, ts TEXT, tray_id TEXT, rule TEXT, severity TEXT, evidence_ids TEXT, threshold_tag TEXT);"
    )
    old.commit()
    old.close()
    monkeypatch.setenv("DB_PATH", str(p))
    conn = deps.open_db()  # an existing file used to skip init_db and crash on the first query
    cols = lambda t: {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}
    assert "updated_at" in cols("assessments") and "pots_json" in cols("masks") and "fault_hash" in cols("frames")
    assert {"detail", "status", "first_ts", "last_ts"} <= cols("alerts")
    conn.execute("SELECT updated_at FROM assessments")
    conn.close()
    assert TestClient(app).get("/trays").status_code == 200


# ------------------------------------------------------------------ 4. validator
def test_overview_does_not_offer_the_latest_frame_as_citable(db):  # noqa: F811
    res, ids = run_tool(ToolContext(db, "asmt_o", AS), "get_tray_overview", {"tray_id": "B"})
    assert ids and not any(re.fullmatch(r"B:\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", i) for i in ids)
    frame = [x for x in res["latest"] if x["metric"] == "latest_frame"]
    assert frame and "id" not in frame[0] and frame[0]["frame_id"].startswith("B:")
    _logged(db, "v_o", ids)
    _propose(db, "v_o")
    cite = final_json(findings=[{"claim": "x", "evidence_ids": [frame[0]["frame_id"]], "confidence": 0.5}])
    assert any(frame[0]["frame_id"] in e for e in validate(cite, "v_o", ALERT_ROW, db))


def test_validator_status_floor_per_alert_severity(db):  # noqa: F811
    _logged(db, "v_s", [SOIL])
    _propose(db, "v_s")
    nominal = final_json(status="nominal")
    assert any(e.startswith("status 'nominal' is below 'watch'") for e in validate(nominal, "v_s", ALERT_ROW, db))
    for ok in ("watch", "alert"):  # WILT_RISK is an `alert`: `watch` is the lowest acceptable status
        assert not any(e.startswith("status") for e in validate(final_json(status=ok), "v_s", ALERT_ROW, db))
    air = db.execute("SELECT id, last_ts FROM alerts WHERE tray_id = 'air'").fetchone()
    own = {"id": air["id"], "rule": "SENSOR_FAULT", "tray_id": "air", "ts": air["last_ts"]}  # severity warning
    base = {"tray_id": "air", "causes_considered": ["sensor_fault"], "recommended_actions": []}
    assert any(e.startswith("status") for e in validate(final_json(status="nominal", **base), "v_s", own, db))
    assert not any(e.startswith("status") for e in validate(final_json(status="watch", **base), "v_s", own, db))


def test_validator_matches_action_magnitude_when_given(db):  # noqa: F811
    _logged(db, "v_m", [SOIL])
    _propose(db, "v_m", magnitude="Moderate  (200 ml)")

    def acts(mag):
        return final_json(recommended_actions=[{"action": "Water tray B", "magnitude": mag, "rationale": "", "evidence_ids": [SOIL]}])

    assert not any(e.startswith("action:") for e in validate(acts("moderate (200 ML)"), "v_m", ALERT_ROW, db))
    assert any(e.startswith("action:") for e in validate(acts("500 ml"), "v_m", ALERT_ROW, db))  # different magnitude
    assert not any(e.startswith("action:") for e in validate(acts(""), "v_m", ALERT_ROW, db))  # none given: action only


def test_validator_sensor_fault_requirement_covers_the_72h_window(db):  # noqa: F811
    _logged(db, "v_w", [SOIL])
    _propose(db, "v_w")
    a = final_json(causes_considered=["underwatering", "low_light"])

    def sensor_errs(as_of):
        return [e for e in validate(a, "v_w", ALERT_ROW, db, as_of=as_of) if "sensor_fault" in e]

    assert sensor_errs(AS)  # open at as_of
    assert sensor_errs(AS + timedelta(hours=5))  # resolved (older than 2 h) but inside the 72 h tool window
    assert sensor_errs(AS + timedelta(hours=70))
    assert not sensor_errs(AS + timedelta(hours=80))  # the episode left the window


def test_likelihood_sum_tolerance_is_1e_9(db):  # noqa: F811
    _logged(db, "v_l", [SOIL])
    _propose(db, "v_l")

    def causes(extra):
        return final_json(probable_causes=[
            {"cause": "underwatering", "likelihood": 0.7, "evidence_ids": [SOIL], "discriminating_check": "c"},
            {"cause": "sensor_fault", "likelihood": 0.3 + extra, "evidence_ids": [SOIL], "discriminating_check": "c"}])

    assert not any("sum to" in e for e in validate(causes(5e-10), "v_l", ALERT_ROW, db))
    assert any("sum to" in e for e in validate(causes(1e-8), "v_l", ALERT_ROW, db))


# ------------------------------------------------------------------ 5. fault disclosure
def test_treatment_and_badge_are_undisclosed_before_fault_start(api):  # noqa: F811
    def trays(as_of):
        return {t["id"]: t for t in api.get("/trays", params={"as_of": as_of}).json()["items"]}

    early = trays("2016-01-04T00:00:00Z")  # before B (01-05) and L (01-08); C started 01-03
    for tid in ("B", "L"):
        assert early[tid]["treatment"] == "undisclosed" and early[tid]["badge"] == "DATASET" and early[tid]["fault_start"] is None
    assert early["A"]["treatment"] == "control" and early["A"]["badge"] == "DATASET"
    assert early["C"]["treatment"] == "low_light" and early["C"]["badge"].startswith("SIMULATED")
    late = trays(AS_OF)
    assert late["B"]["treatment"] == "no_water" and late["B"]["badge"] == "SIMULATED FAULT (image transform)"
    assert late["L"]["treatment"] == "undisclosed"  # L starts 01-08, after AS_OF
    d_early = api.get("/trays/B", params={"as_of": "2016-01-04T00:00:00Z"}).json()
    assert d_early["treatment"] == "undisclosed" and d_early["badge"] == "DATASET"
    d_late = api.get("/trays/B", params={"as_of": AS_OF}).json()
    assert d_late["treatment"] == "no_water" and d_late["badge"] == "SIMULATED FAULT (image transform)"


# ------------------------------------------------------------------ 6. wilt ghosts
P = FaultParams(shrink_per_day=0.05, s_min=0.55, desat_per_day=0.08)
SOIL_BGR = (10, 15, 20)


def test_wilt_erases_every_current_plant_pixel_outside_the_roi_and_its_fringe():
    ref = np.full((160, 160, 3), SOIL_BGR, np.uint8)
    cur = ref.copy()
    cur[40:80, 130:150] = (30, 170, 60)  # a growing leaf OUTSIDE the ROI (x >= 100) and the margin
    cur[100:120, 20:40] = (30, 170, 60)  # a plant inside the ROI, absent from the reference
    cv2.rectangle(cur, (17, 97), (42, 122), (25, 60, 30), 3)  # dark-green fringe 2-4 px out that the HSV mask misses
    ref[40:80, 130:150] = (30, 170, 60)  # the reference also has a leaf out there (not inpainted when ROI-clipped)
    assert not plant_mask(cur)[97:98, 17:43].any()  # the fringe is not in the plant mask itself
    out = wilt_transform(cur, 2.0, P, (0, 0, 100, 100), ref, margin=10)
    assert not plant_mask(out).any()  # nothing of either frame's plants survives outside the shrunk region
    soil = np.full_like(out, SOIL_BGR)
    assert np.abs(out.astype(int) - soil.astype(int)).max() <= 6  # clean soil (inpaint rounding), fringe (diff 45) included


def test_wilt_erases_dim_leaf_parts_shaded_below_the_plant_threshold():
    ref = np.full((160, 160, 3), SOIL_BGR, np.uint8)
    cur = ref.copy()
    cur[88:132, 8:52] = (20, 50, 25)  # dark-green leaf tips (shade under a pot rim): V = 50 is below the plant threshold
    cur[100:120, 20:40] = (30, 170, 60)  # the bright part, 12 px inside the dim part's edge
    assert plant_mask(cur)[88:132, 8:52].sum() == 400  # only the bright part is a "plant" pixel
    out = wilt_transform(cur, 2.0, P, (0, 0, 160, 160), ref, margin=0)
    assert np.abs(out.astype(int) - np.full_like(out, SOIL_BGR).astype(int)).max() <= 6


def test_wilt_still_shrinks_reference_plants_inside_roi_plus_margin():
    ref = np.full((160, 160, 3), SOIL_BGR, np.uint8)
    ref[40:80, 40:80] = (30, 170, 60)
    ref[40:80, 104:116] = (30, 170, 60)  # outside ROI (x < 100) but inside the 20 px margin
    out = wilt_transform(ref, 4.0, P, (0, 0, 100, 100), ref, margin=20)
    s = 1 - 0.05 * 4
    assert plant_mask(out)[40:80, 100:130].sum() == pytest.approx(12 * 40 * s * s, rel=0.3)
    assert plant_mask(out)[40:80, 40:80].sum() == pytest.approx(1600 * s * s, rel=0.15)


def test_faults_version_bump_regenerates_b_but_not_c():
    assert FAULTS_VERSION == "wilt-4"
    assert fault_hash(B) != "68d47e8271be5cb8"  # the stored wilt-3 hash of tray B: ingest regenerates its frames
    assert fault_hash(C) == "f39e9ea9fb595718"  # tray C's hash depends only on its own warp (TIMEWARP_VERSION)
    assert fault_hash(A) is None
    assert fault_hash(B.model_copy(update={"roi_margin_px": 10})) != fault_hash(B)


# ------------------------------------------------------------------ 7. recompute from cache
def _segmented(tmp_path):
    conn = init_db(":memory:")
    for h in (9, 10, 11):
        for tray in ("A", "B"):
            _frame(conn, tmp_path, tray, f"2016-01-06T{h:02d}:00:00Z", 60 - 2 * h)
    conn.commit()
    stats = segment_frames(conn, CFG, HsvBackend(), cache_dir=tmp_path / "cache")
    assert stats["segmented"] == 6
    return conn


def _masks(conn):
    return [tuple(r) for r in conn.execute("SELECT id, plant_id, area_px, exg, chlorosis_frac, rle, pots_json FROM masks ORDER BY id")]


def test_recompute_from_cache_rebuilds_masks_with_no_backend_call(tmp_path, monkeypatch):
    conn = _segmented(tmp_path)
    want = _masks(conn)
    assert any(r[6] for r in want) and any(r[1] for r in want)
    conn.execute("DELETE FROM masks")
    conn.execute("DELETE FROM plants")

    async def boom(*a, **k):
        raise AssertionError("a model/backend call was made")

    monkeypatch.setattr(HsvBackend, "acquire", boom)
    stats = recompute_from_cache(conn, CFG, "hsv", cache_dir=tmp_path / "cache")
    assert stats["recomputed"] == 6 and _masks(conn) == want


def test_recompute_from_cache_refuses_when_any_frame_is_uncached(tmp_path):
    conn = _segmented(tmp_path)
    before = _masks(conn)
    next((tmp_path / "cache").glob("*.json")).unlink()
    with pytest.raises(CacheMiss):
        recompute_from_cache(conn, CFG, "hsv", cache_dir=tmp_path / "cache")
    assert _masks(conn) == before  # nothing was written


def test_segment_all_has_the_recompute_flag_and_refuses(tmp_path, monkeypatch):
    import segment_all

    monkeypatch.setattr(sys, "argv", ["segment_all.py", "--recompute-from-cache", "--db", str(tmp_path / "x.db"), "--backend", "hsv"])
    conn = init_db(tmp_path / "x.db")
    _frame(conn, tmp_path, "A", "2016-01-06T10:00:00Z", 40)
    conn.commit()
    conn.close()
    with pytest.raises(SystemExit) as e:
        segment_all.main()
    assert "refusing" in str(e.value)


# ------------------------------------------------------------------ 8. cache key, lamp
def _img(path: Path, v: int) -> Path:
    cv2.imwrite(str(path), np.full((16, 16, 3), v, np.uint8))
    return path


def test_cache_key_is_raw_sha_plus_fault_hash_not_derived_bytes(tmp_path):
    raw, d1, d2 = _img(tmp_path / "raw.jpg", 50), _img(tmp_path / "d1.jpg", 60), _img(tmp_path / "d2.jpg", 61)
    k = cache_key(d1, "plant", "api", raw, "h1")
    assert cache_key(d2, "plant", "api", raw, "h1") == k  # a re-encoded derived frame keeps its key
    assert cache_key(d1, "plant", "api", raw, "h2") != k  # a changed fault does not
    assert cache_key(d1, "plant", "hsv", raw, "h1") != k and cache_key(d1, "other", "api", raw, "h1") != k
    assert cache_key(raw, "plant", "api") == legacy_cache_key(raw, "plant", "api")  # dataset frames need no migration


def test_migrate_cache_copies_legacy_entries_so_nothing_rebills(tmp_path, monkeypatch):
    raw_dir, cache = tmp_path / "raw", tmp_path / "cache"
    raw_dir.mkdir()
    cache.mkdir()
    conn = init_db(":memory:")
    for name, fh in (("B_a.jpg", "h_old"), ("B_b.jpg", "h_new")):
        _img(raw_dir / name, 50)
        _img(tmp_path / name, 70 if name == "B_a.jpg" else 71)  # the derived frame
        conn.execute(
            "INSERT INTO frames(id, tray_id, ts, path, cm2_per_px, calib_flag, source, fault_hash) VALUES (?,?,?,?,0.01,'n','fault_transform',?)",
            (f"B:{name}", "B", "2016-01-06T10:00:00Z", str(tmp_path / name), fh),
        )
    conn.commit()
    monkeypatch.setattr(segment, "RAW_DIR", raw_dir)
    frames = conn.execute("SELECT id, tray_id, ts, path, cm2_per_px, fault_hash FROM frames ORDER BY id").fetchall()
    old = cache / f"{legacy_cache_key(tmp_path / 'B_a.jpg', 'plant', 'api')}.json"
    old.write_text(json.dumps({"payload": "[]"}), encoding="utf-8")  # frame a was billed under the old key
    assert estimate_cost_usd(conn, frames, cache) > 0
    assert migrate_cache(conn, cache)["copied"] == 1
    assert estimate_cost_usd(conn, frames[:1], cache) == 0  # frame a: copied, nothing to bill
    assert estimate_cost_usd(conn, frames, cache) > 0  # frame b was never cached
    assert migrate_cache(conn, cache)["copied"] == 0 and old.exists()  # idempotent; the old file stays


@pytest.mark.parametrize(
    ("on", "off", "hhmm", "want"),
    [
        ("08:00", "22:00", "07:59", False), ("08:00", "22:00", "08:00", True), ("08:00", "22:00", "21:59", True),
        ("08:00", "22:00", "22:00", False),
        ("20:00", "06:00", "23:30", True), ("20:00", "06:00", "00:00", True), ("20:00", "06:00", "05:59", True),
        ("20:00", "06:00", "06:00", False), ("20:00", "06:00", "12:00", False), ("20:00", "06:00", "19:59", False),
    ],
)
def test_lamp_is_on_handles_overnight_schedules(on, off, hhmm, want):
    from backend.config import Lamp

    h, m = map(int, hhmm.split(":"))
    assert lamp_is_on(Lamp(on=on, off=off), datetime(2016, 1, 6, h, m, tzinfo=UTC)) is want
