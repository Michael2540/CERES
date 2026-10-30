"""Fill plants, masks.plant_id and the canopy rows' pots_json from the masks already in the DB.

No SAM calls and no image reads: each `plant` mask row is decoded from its RLE, assigned to the pot that
holds its centroid (backend/perception/pots.py), and the canopy row gets the per-pot union area. Idempotent.
Masks keep whatever ROI clip they were stored with; use `segment_all.py --recompute-from-cache` to re-clip.

    python scripts/backfill_pots.py [--db data/ceres.db] [--trays A,B]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import TraysConfig, load_settings, load_trays
from backend.db import init_db
from backend.perception import pots
from backend.perception.metrics import rle_decode


def backfill(conn: sqlite3.Connection, cfg: TraysConfig, trays: list[str] | None = None) -> dict:
    by_id = {t.id: t for t in cfg.trays}
    q = (
        "SELECT m.id, m.frame_id, m.tray_id, f.ts FROM masks m JOIN frames f ON f.id = m.frame_id"
        " WHERE m.kind = 'canopy'"
    )
    canopy = [r for r in conn.execute(q + " ORDER BY f.ts").fetchall() if not trays or r["tray_id"] in trays]
    stats = {"frames": len(canopy), "plant_rows": 0, "plants": 0}
    for c in canopy:
        tray = by_id[c["tray_id"]]
        rows = conn.execute(
            "SELECT id, rle FROM masks WHERE frame_id = ? AND kind = 'plant' ORDER BY id", (c["frame_id"],)
        ).fetchall()
        pot_ids, pot_area = pots.assign(tray, [rle_decode(json.loads(r["rle"])) for r in rows])
        held = pots.holding(pot_area)
        for pid in held:
            conn.execute(
                "INSERT INTO plants(id, tray_id, first_seen_ts, label) VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE"
                " SET first_seen_ts = MIN(first_seen_ts, excluded.first_seen_ts)",
                (pid, c["tray_id"], c["ts"], f"pot {pid.rsplit('pot', 1)[1]}"),
            )
        for r, pid in zip(rows, pot_ids):
            conn.execute("UPDATE masks SET plant_id = ? WHERE id = ?", (pid if pid in held else None, r["id"]))
        conn.execute("UPDATE masks SET pots_json = ? WHERE id = ?", (json.dumps(pot_area), c["id"]))
        stats["plant_rows"] += len(rows)
    stats["plants"] = conn.execute("SELECT COUNT(*) FROM plants").fetchone()[0]
    conn.commit()
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(ROOT / load_settings().db_path))
    ap.add_argument("--trays", help="comma-separated tray ids")
    a = ap.parse_args(argv)
    conn = init_db(a.db)
    print(backfill(conn, load_trays(), a.trays.split(",") if a.trays else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
