"""Tray C time-warp over the union of mission frames (data/raw/aled) and extra sources (data/raw/aled_src)."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.config import load_trays
from backend.db import init_db
from backend.perception import segment
from backend.perception.ingest import fault_hash, ingest_folder
from backend.perception.sam_client import cache_key
from backend.perception.segment import MissingRaw, frame_source
from backend.sim.faults import timewarp_source_ts
from backend.timeutil import lamp_is_on

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_dataset import timewarp_source_jobs

CFG = load_trays()
C = next(t for t in CFG.trays if t.id == "C")
START = datetime.fromisoformat(C.fault_start)
GF = C.fault.growth_factor
DERIVED = "SELECT COUNT(*) FROM frames WHERE fault_hash IS NOT NULL"


def _name(tray: str, ts: datetime) -> str:
    return f"{tray}_{ts.strftime('%Y%m%dT%H%M%S')}.jpg"


def _write(path: Path, k: int) -> None:
    img = np.full((40, 50, 3), (40, 50, 70), np.uint8)
    img[10:30, 10:40] = (30, 170, 60)
    img[:4, : 2 + k % 40] = (200, 200, 200)  # distinct bytes per frame
    cv2.imwrite(str(path), img)


def _pools(tmp_path: Path, extra_step_min: int | None):
    mission, src = tmp_path / "aled", tmp_path / "aled_src"
    mission.mkdir()
    src.mkdir()
    ts = [START - timedelta(hours=4) + timedelta(hours=2 * i) for i in range(60)]
    ts = [t for t in ts if lamp_is_on(C.lamp, t)][:40]  # mission frames exist only in imaging hours, as in ALED
    for i, t in enumerate(ts):
        _write(mission / _name("C", t), i)
    if extra_step_min:
        t, k = START, 0
        while t < ts[-1]:
            if t not in ts:
                k += 1
                _write(src / _name("C", t), 100 + k)
            t += timedelta(minutes=extra_step_min)
    return mission, src, ts


def test_nearest_over_union_pool():
    mission = [START + timedelta(hours=h) for h in (0, 2, 4, 6)]
    extra = [START + timedelta(hours=1, minutes=m) for m in (0, 13, 26)]
    t = START + timedelta(hours=4)  # target = START + 1.6 h
    assert timewarp_source_ts(t, START, GF, mission) == START + timedelta(hours=2)
    assert timewarp_source_ts(t, START, GF, sorted(mission + extra)) == START + timedelta(hours=1, minutes=26)
    later = START + timedelta(hours=1, minutes=1)  # nearer to the target but after ts: never chosen
    one_h = START + timedelta(hours=1)
    assert timewarp_source_ts(one_h, START, 0.9, [*mission, later]) <= one_h


def test_fetch_jobs_pick_nearest_archive_frame_and_skip_mission_frames():
    mission = [START + timedelta(hours=h) for h in (0, 2, 4)]
    day = START.replace(hour=9, minute=0)
    arch = [day + timedelta(minutes=13 * i) for i in range(40)]  # 09:00 .. 17:27
    names = [
        f"images_and_annotations/PSI_Tray033/tv/PSI_Tray033_{t.strftime('%Y-%m-%d--%H-%M-%S')}_top.png"
        for t in arch
    ]
    jobs = timewarp_source_jobs(names, "C", mission, START, GF)
    assert jobs and all(t not in mission for _, t in jobs)
    target = START + timedelta(hours=4) * GF
    best = min([*arch, *mission], key=lambda a: (abs((a - target).total_seconds()), a))
    assert best in {t for _, t in jobs}
    assert all(9 <= t.hour < 21 for _, t in jobs)  # imaging hours only


def test_mission_frames_unaffected_by_extra_pool(conn, tmp_path):
    mission, src, ts = _pools(tmp_path, 13)
    assert ingest_folder(conn, mission, CFG, tmp_path / "derived", src, tmp_path / "cache") == len(ts)
    ids = {r[0] for r in conn.execute("SELECT id FROM frames")}
    assert ids == {f"C:{t.strftime('%Y-%m-%dT%H:%M:%SZ')}" for t in ts}  # no aled_src frame became a frame
    assert len(list((tmp_path / "derived").glob("C_*.jpg"))) == sum(t >= START for t in ts)


def test_consecutive_post_fault_frames_get_distinct_sources_with_dense_pool(conn, tmp_path):
    mission, src, _ = _pools(tmp_path, 13)
    ingest_folder(conn, mission, CFG, tmp_path / "derived", src, tmp_path / "cache")
    names = [r[0] for r in conn.execute("SELECT src_file FROM frames WHERE fault_hash IS NOT NULL ORDER BY ts")]
    assert len(names) > 10 and all(names) and len(set(names)) == len(names)
    conn2 = init_db(":memory:")  # the 2 h mission pool alone repeats sources
    ingest_folder(conn2, mission, CFG, tmp_path / "d2", tmp_path / "none", tmp_path / "cache")
    old = [r[0] for r in conn2.execute("SELECT src_file FROM frames WHERE fault_hash IS NOT NULL ORDER BY ts")]
    assert len(set(old)) < len(old)


def test_legacy_rows_keep_masks_when_source_unchanged_and_lose_them_when_changed(conn, tmp_path):
    mission, src, ts = _pools(tmp_path, None)
    ingest_folder(conn, mission, CFG, tmp_path / "derived", src, tmp_path / "cache")
    conn.execute("UPDATE frames SET src_file = NULL")  # rows written before aled_src existed
    conn.execute(
        "INSERT INTO masks(id, frame_id, tray_id, kind) SELECT id || ':canopy', id, tray_id, 'canopy' FROM frames"
    )
    conn.commit()
    ingest_folder(conn, mission, CFG, tmp_path / "derived", src, tmp_path / "cache")
    assert conn.execute("SELECT COUNT(*) FROM masks").fetchone()[0] == len(ts)  # unchanged: every mask stays
    assert conn.execute(DERIVED + " AND src_file IS NULL").fetchone()[0] == 0
    for i in range(1, 400):  # a dense source pool appears
        t = START + timedelta(minutes=13 * i + 1)
        if t not in ts:
            _write(src / _name("C", t), i)
    conn.execute("UPDATE frames SET src_file = NULL")
    ingest_folder(conn, mission, CFG, tmp_path / "derived", src, tmp_path / "cache")
    left = conn.execute("SELECT COUNT(*) FROM masks").fetchone()[0]
    assert 0 < left < len(ts)  # only frames whose nearest source changed were regenerated


def test_frame_source_points_at_actual_source_and_missing_raises(conn, tmp_path):
    mission, src, _ = _pools(tmp_path, 13)
    ingest_folder(conn, mission, CFG, tmp_path / "derived", src, tmp_path / "cache")
    q = f"SELECT {segment._COLS} FROM frames f WHERE fault_hash IS NOT NULL ORDER BY ts LIMIT 1 OFFSET 5"
    f = conn.execute(q).fetchone()
    _, raw, fh = frame_source(f, mission)
    own = mission / Path(f["path"]).name
    assert raw.name == f["src_file"] and raw.is_file() and fh == fault_hash(C)
    assert cache_key(f["path"], "plant", "api", raw, fh) != cache_key(f["path"], "plant", "api", own, fh)
    raw.unlink()
    with pytest.raises(MissingRaw):
        frame_source(f, mission)


def test_cache_keys_unchanged_for_a_b_l(tmp_path):
    hashes = {t.id: fault_hash(t) for t in CFG.trays}
    assert hashes == {"A": None, "B": "2a037cf5bdf33bc1", "C": "f39e9ea9fb595718", "L": None}
    conn = init_db(":memory:")
    (tmp_path / "raw").mkdir()
    (tmp_path / "aled_src").mkdir()
    img = tmp_path / "raw" / "B_20160106T100000.jpg"
    _write(img, 1)
    _write(tmp_path / "aled_src" / img.name, 2)  # a same-named extra file must not hijack tray B
    conn.execute(
        "INSERT INTO frames(id, tray_id, ts, path, cm2_per_px, calib_flag, source, fault_hash) VALUES"
        " ('B:2016-01-06T10:00:00Z','B','2016-01-06T10:00:00Z',?,0.01,'n','fault_transform',?)",
        (str(img), hashes["B"]),
    )
    f = conn.execute(f"SELECT {segment._COLS} FROM frames f").fetchone()
    _, raw, fh = frame_source(f, tmp_path / "raw")
    assert raw == img and fh == hashes["B"]
    assert cache_key(img, "plant", "api", raw, fh) == cache_key(img, "plant", "api", img, hashes["B"])
    conn.execute("UPDATE frames SET fault_hash = NULL")  # dataset frames (A, L, pre-fault) key on their own sha
    f = conn.execute(f"SELECT {segment._COLS} FROM frames f").fetchone()
    assert frame_source(f, tmp_path / "raw")[1:] == (None, None)


def test_lamp_hours_warp_targets_lit_time_only():
    """Under the lamp-hours warp a mission frame at 21:00 on day 2 maps to a lit time, never into the night gap."""
    from backend.sim.faults import lit_target

    t = START + timedelta(days=1, hours=13)  # day 2, 21:00 (lamp 08:00-22:00)
    target = lit_target(t, START, GF, C.lamp)
    assert lamp_is_on(C.lamp, target) and START <= target < t
    assert lit_target(START + timedelta(hours=10), START, GF, C.lamp) == START + timedelta(hours=4)  # 0.4 x 10 lit hours
