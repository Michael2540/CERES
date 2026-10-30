"""run_cycle on a tiny synthetic DB: hsv segmentation, expected alerts, dedupe, no data after as_of."""

from __future__ import annotations

from datetime import timedelta

import cv2
import numpy as np

from backend.config import load_trays
from backend.db import init_db
from backend.perception.sam_client import HsvBackend
from backend.pipeline import run_cycle
from backend.timeutil import iso_z, parse_ts

CFG = load_trays()
AS_OF = "2016-01-06T14:00:00Z"
FRAME_HOURS = range(9, 15)  # one frame per hour, lamp on


def _img(side: int) -> np.ndarray:
    img = np.full((768, 1024, 3), (40, 50, 70), np.uint8)
    img[250 : 250 + side, 300 : 300 + side] = (30, 170, 60)  # BGR green rosette, inside pot (1,1) of the 5x4 grid
    return img


def _frame(conn, tmp_path, tray: str, ts: str, side: int) -> None:
    p = tmp_path / f"{tray}_{ts.replace('-', '').replace(':', '')}.jpg"
    cv2.imwrite(str(p), _img(side))
    conn.execute(
        "INSERT INTO frames(id, tray_id, ts, path, cm2_per_px, calib_flag, source) VALUES (?,?,?,?,0.01,'nominal','dataset')",
        (f"{tray}:{ts}", tray, ts, str(p)),
    )


def _reading(conn, tray: str, metric: str, ts: str, value: float, source: str = "simulated") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO sensor_readings(id, ts, tray_id, metric, value, source) VALUES (?,?,?,?,?,?)",
        (f"{tray}:{metric}:{ts}", ts, tray, metric, value, source),
    )


def build(conn, tmp_path, extra: bool = False) -> None:
    """Tray B dries out and shrinks (WILT_RISK); the air RH sensor flatlines for the last hour (SENSOR_FAULT).

    extra=True adds frames and readings AFTER AS_OF that would change the outcome if they were read."""
    for i, h in enumerate(FRAME_HOURS):
        ts = f"2016-01-06T{h:02d}:00:00Z"
        _frame(conn, tmp_path, "B", ts, 60 - 2 * i)
        _frame(conn, tmp_path, "A", ts, 60)
    t, end = parse_ts("2016-01-03T08:00:00Z"), parse_ts(AS_OF)
    k = 0
    while t <= end:
        ts = iso_z(t)
        wet = t < parse_ts("2016-01-05T08:00:00Z")  # tray B fault_start
        _reading(conn, "B", "soil_vwc", ts, (0.30 if wet else 0.10) + 0.01 * (k % 3))
        _reading(conn, "air", "temp_c", ts, 22.0 + 0.1 * (k % 4))
        _reading(conn, "air", "rh_pct", ts, 55.0 if t >= end - timedelta(minutes=70) else 50.0 + (k % 5))
        _reading(conn, "B", "lamp_on", ts, 1.0, "derived")
        t += timedelta(minutes=10)
        k += 1
    if extra:
        for h in range(15, 21):
            ts = f"2016-01-06T{h:02d}:00:00Z"
            _frame(conn, tmp_path, "B", ts, 5)
            _frame(conn, tmp_path, "A", ts, 5)
        for m in range(10, 600, 10):
            ts = iso_z(parse_ts(AS_OF) + timedelta(minutes=m))
            _reading(conn, "B", "soil_vwc", ts, 0.9)
            _reading(conn, "air", "rh_pct", ts, 120.0)  # out of range: would be another SENSOR_FAULT
    conn.commit()


def snapshot(conn) -> list[tuple]:
    return [
        tuple(r)
        for r in conn.execute(
            "SELECT id, last_ts, tray_id, rule, severity, evidence_ids, threshold_tag, detail, status FROM alerts"
            " ORDER BY id"
        )
    ]


def cycle(conn, tmp_path, as_of: str = AS_OF, **kw):
    return run_cycle(
        conn, CFG, as_of, HsvBackend(), ingest=False, cache_dir=tmp_path / "cache", **kw
    )


def test_run_cycle_segments_and_raises_expected_alerts(tmp_path):
    conn = init_db(tmp_path / "t.db")
    build(conn, tmp_path)
    res = cycle(conn, tmp_path)
    assert res.segmentation["segmented"] == 12 and res.segmentation["errors"] == 0
    assert conn.execute("SELECT COUNT(*) FROM masks WHERE kind = 'canopy'").fetchone()[0] == 12
    assert conn.execute("SELECT COUNT(*) FROM trays").fetchone()[0] == len(CFG.trays)
    rows = {(r[3], r[2]): r for r in snapshot(conn)}
    assert set(rows) == {("WILT_RISK", "B"), ("SENSOR_FAULT", "air")}  # nothing on A, C, L
    wilt, fault = rows[("WILT_RISK", "B")], rows[("SENSOR_FAULT", "air")]
    assert wilt[1] == AS_OF and wilt[4] == "alert" and wilt[8] == "open"
    assert wilt[0] == "alert_B_WILT_RISK_20160106T140000Z"
    assert "B:area_cm2:2016-01-06T09:00:00Z" in wilt[5]  # the canopy peak, in the series' id format
    assert fault[4] == "warning" and fault[7].startswith("rh_pct flatline")
    assert res.new_alerts and not res.updated_alerts


def test_dedupe_updates_instead_of_inserting(tmp_path):
    conn = init_db(tmp_path / "t.db")
    build(conn, tmp_path)
    cycle(conn, tmp_path)
    first = snapshot(conn)
    again = cycle(conn, tmp_path)  # same as_of: pure update
    assert snapshot(conn) == first and not again.new_alerts and len(again.updated_alerts) == 2

    # one hour later the same rule/tray updates the alert's ts instead of adding a second row
    later = iso_z(parse_ts(AS_OF) + timedelta(hours=1))
    _frame(conn, tmp_path, "B", later, 46)
    _frame(conn, tmp_path, "A", later, 60)
    for m in range(10, 70, 10):
        ts = iso_z(parse_ts(AS_OF) + timedelta(minutes=m))
        _reading(conn, "B", "soil_vwc", ts, 0.10)
        _reading(conn, "air", "temp_c", ts, 22.0 + 0.1 * (m % 4))
        _reading(conn, "air", "rh_pct", ts, 55.0)  # still flat
        _reading(conn, "B", "lamp_on", ts, 1.0, "derived")
    conn.commit()
    res = cycle(conn, tmp_path, later)
    rows = snapshot(conn)
    assert len(rows) == 2 and not res.new_alerts
    wilt = next(r for r in rows if r[3] == "WILT_RISK")
    assert wilt[0] == "alert_B_WILT_RISK_20160106T140000Z" and wilt[1] == later  # id kept, ts moved
    fault = next(r for r in rows if r[3] == "SENSOR_FAULT")
    assert fault[1] == later and "..2016-01-06T15:00:00Z" in fault[7]  # same fault, extended span


def test_alert_resolves_when_it_stops_firing(tmp_path):
    conn = init_db(tmp_path / "t.db")
    build(conn, tmp_path)
    cycle(conn, tmp_path)
    # two days later (no new frames; the sensors went silent) the old alerts no longer fire and are resolved
    cycle(conn, tmp_path, "2016-01-08T14:00:00Z")
    status = {r[0]: r[8] for r in snapshot(conn)}
    assert status["alert_B_WILT_RISK_20160106T140000Z"] == "resolved"
    assert [r[0] for r in snapshot(conn) if r[8] == "open"] and all(
        r[3] == "SENSOR_FAULT" and "gap" in r[7] for r in snapshot(conn) if r[8] == "open"
    )  # only the new data-gap faults are open


def test_no_data_after_as_of_is_read(tmp_path):
    a = init_db(tmp_path / "a.db")
    (tmp_path / "a_imgs").mkdir()
    build(a, tmp_path / "a_imgs")
    b = init_db(tmp_path / "b.db")
    (tmp_path / "b_imgs").mkdir()
    build(b, tmp_path / "b_imgs", extra=True)
    cycle(a, tmp_path)
    res = cycle(b, tmp_path)
    assert snapshot(a) == snapshot(b)
    # frames after as_of stay unsegmented
    n_future = b.execute("SELECT COUNT(*) FROM frames WHERE ts > ?", (AS_OF,)).fetchone()[0]
    assert n_future == 12
    assert b.execute("SELECT COUNT(*) FROM masks WHERE frame_id IN (SELECT id FROM frames WHERE ts > ?)", (AS_OF,)).fetchone()[0] == 0
    assert res.segmentation["segmented"] == 12


def _extend(conn, tmp_path) -> None:
    """Frames and readings for 15:00 and 16:00 so the episodes keep going after AS_OF."""
    for i, h in enumerate((15, 16)):
        ts = f"2016-01-06T{h:02d}:00:00Z"
        _frame(conn, tmp_path, "B", ts, 48 - 2 * i)
        _frame(conn, tmp_path, "A", ts, 60)
    for m in range(10, 130, 10):
        ts = iso_z(parse_ts(AS_OF) + timedelta(minutes=m))
        _reading(conn, "B", "soil_vwc", ts, 0.10 + 0.01 * (m % 30) / 10)
        _reading(conn, "air", "temp_c", ts, 22.0 + 0.1 * (m % 40) / 10)
        _reading(conn, "air", "rh_pct", ts, 55.0)
        _reading(conn, "B", "lamp_on", ts, 1.0, "derived")
    conn.commit()


def test_alerts_have_no_time_travel(tmp_path):
    """Replay to 16:00, then ask at earlier as_of values: same answer as a replay that stopped there."""
    from backend.api.alerts import visible_alerts

    (tmp_path / "full").mkdir()
    full = init_db(tmp_path / "full.db")
    build(full, tmp_path / "full")
    _extend(full, tmp_path / "full")
    for h in range(10, 17):
        cycle(full, tmp_path, f"2016-01-06T{h:02d}:00:00Z")
    assert len(full.execute("SELECT * FROM alerts WHERE rule = 'WILT_RISK'").fetchall()) == 1  # one episode
    n_events = full.execute("SELECT COUNT(*) FROM alert_events WHERE alert_id LIKE '%WILT%'").fetchone()[0]
    assert n_events >= 4
    seen_wilt = False
    for h in (10, 11, 12, 14, 16):
        t = f"2016-01-06T{h:02d}:00:00Z"
        (tmp_path / f"stop{h}").mkdir()
        stop = init_db(tmp_path / f"stop{h}.db")
        build(stop, tmp_path / f"stop{h}")
        _extend(stop, tmp_path / f"stop{h}")  # same data: only the cycles differ
        for k in range(10, h + 1):
            cycle(stop, tmp_path, f"2016-01-06T{k:02d}:00:00Z")
        want = visible_alerts(stop, parse_ts(t))
        got = visible_alerts(full, parse_ts(t))
        assert got == want, (t, got, want)
        for a in got:
            assert a["ts"] <= t
            for eid in a["evidence_ids"]:  # evidence ids never name a later timestamp
                assert eid[-20:] <= t, (eid, t)
        seen_wilt |= any(a["rule"] == "WILT_RISK" for a in got)
    assert seen_wilt
    # the episode starts when the rule first fired and is invisible before that
    first = full.execute("SELECT first_ts FROM alerts WHERE rule = 'WILT_RISK'").fetchone()[0]
    assert visible_alerts(full, parse_ts(first) - timedelta(seconds=1)) == []


def test_persistent_out_of_range_fault_is_one_episode(tmp_path):
    conn = init_db(":memory:")
    base = parse_ts("2016-01-06T00:00:00Z")
    for k in range(15 * 6 + 1):  # rh_pct = 150 every 10 min over 15 h
        _reading(conn, "air", "rh_pct", iso_z(base + timedelta(minutes=10 * k)), 150.0)
    conn.commit()
    for h in range(1, 15):  # 14 hourly cycles
        run_cycle(conn, CFG, iso_z(base + timedelta(hours=h)), None, ingest=False, segment=False, cache_dir=tmp_path)
    rows = conn.execute("SELECT id, first_ts, last_ts FROM alerts WHERE rule = 'SENSOR_FAULT'").fetchall()
    assert len(rows) == 1, [tuple(r) for r in rows]
    assert rows[0]["last_ts"] > rows[0]["first_ts"]
    n = conn.execute("SELECT COUNT(*) FROM alert_events WHERE alert_id = ?", (rows[0]["id"],)).fetchone()[0]
    assert n == 14


def test_fault_key_separates_types_and_ignores_span():
    from backend.pipeline import _fault_key

    assert _fault_key("rh_pct out of range A..B") == _fault_key("rh_pct out of range C..D") == "rh_pct range"
    assert _fault_key("rh_pct flatline A..B") != _fault_key("rh_pct gap A..B")
    assert _fault_key("rh_pct gap A..B") != _fault_key("temp_c gap A..B")
