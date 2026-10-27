import cv2
import numpy as np

from backend.config import load_trays
from backend.perception.ingest import ingest_folder


def _img(k: int = 0) -> np.ndarray:
    img = np.full((100, 120, 3), (40, 50, 70), np.uint8)
    img[30:70, 30:90] = (30, 170, 60)
    img[:10, : 10 + 10 * k] = (200, 200, 200)  # makes every file distinct
    return img


def _folder(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    for i, name in enumerate(
        [
            "A_20151214T125406",
            "A_20160106T090000",
            "B_20160104T090000",
            "B_20160106T090000",
            "C_20160102T090000",
            "C_20160104T090000",
            "C_20160105T090000",
            "L_20160109T090000",
        ]
    ):
        cv2.imwrite(str(d / f"{name}.jpg"), _img(i))
    cv2.imwrite(str(d / "A_20151214T125406_mask.png"), np.zeros((5, 5), np.uint8))
    return d


def test_ingest_sources_and_idempotent(conn, tmp_path):
    cfg, d, der = load_trays(), _folder(tmp_path), tmp_path / "derived"
    assert ingest_folder(conn, d, cfg, der) == 8
    assert ingest_folder(conn, d, cfg, der) == 0
    rows = {r["id"]: r for r in conn.execute("SELECT * FROM frames")}
    assert rows["A:2015-12-14T12:54:06Z"]["source"] == "dataset"
    assert rows["A:2016-01-06T09:00:00Z"]["source"] == "dataset"  # tray A has no fault
    assert rows["B:2016-01-04T09:00:00Z"]["source"] == "dataset"  # before fault_start 01-05T08
    assert rows["B:2016-01-06T09:00:00Z"]["source"] == "fault_transform"
    assert rows["C:2016-01-02T09:00:00Z"]["source"] == "dataset"
    assert rows["C:2016-01-04T09:00:00Z"]["source"] == "fault_timewarp"
    assert rows["L:2016-01-09T09:00:00Z"]["source"] == "dataset"  # L: sensor-only fault
    r = rows["B:2016-01-06T09:00:00Z"]
    assert (
        r["path"].endswith("B_20160106T090000.jpg")
        and cv2.imread(str(der / "B_20160106T090000.jpg")) is not None
    )
    assert r["cm2_per_px"] == 0.01 and r["calib_flag"] == "nominal"
    assert len(rows) == 8  # *_mask.png skipped


def test_stale_derived_frames_are_regenerated(conn, tmp_path):
    cfg, d, der = load_trays(), _folder(tmp_path), tmp_path / "derived"
    ingest_folder(conn, d, cfg, der)
    fid = "B:2016-01-06T09:00:00Z"
    h0 = conn.execute("SELECT fault_hash FROM frames WHERE id = ?", (fid,)).fetchone()[0]
    assert (
        h0
        and conn.execute("SELECT fault_hash FROM frames WHERE id = 'A:2016-01-06T09:00:00Z'").fetchone()[0]
        is None
    )
    conn.execute("INSERT INTO masks(id, frame_id, tray_id, kind) VALUES ('m', ?, 'B', 'canopy')", (fid,))
    conn.commit()
    before = cv2.imread(str(der / "B_20160106T090000.jpg")).copy()
    assert ingest_folder(conn, d, cfg, der) == 0  # same params: untouched, masks kept
    assert conn.execute("SELECT COUNT(*) FROM masks").fetchone()[0] == 1
    cfg.by_id("B").fault.shrink_per_day = 0.2  # parameters changed
    assert ingest_folder(conn, d, cfg, der) == 0
    row = conn.execute("SELECT fault_hash, source FROM frames WHERE id = ?", (fid,)).fetchone()
    assert row[0] != h0 and row[1] == "fault_transform"
    assert conn.execute("SELECT COUNT(*) FROM masks").fetchone()[0] == 0
    assert (cv2.imread(str(der / "B_20160106T090000.jpg")) != before).any()
    assert conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0] == 8  # no duplicates


def test_init_db_adds_fault_hash_to_old_db(tmp_path):
    import sqlite3

    from backend.db import init_db

    p = tmp_path / "old.db"
    c = sqlite3.connect(p)
    c.execute(
        "CREATE TABLE frames(id TEXT PRIMARY KEY, tray_id TEXT, ts TEXT, path TEXT, cm2_per_px REAL, calib_flag TEXT, source TEXT)"
    )
    c.commit()
    c.close()
    conn = init_db(p)
    assert "fault_hash" in {r[1] for r in conn.execute("PRAGMA table_info(frames)")}
    init_db(p)  # idempotent
