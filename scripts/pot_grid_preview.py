"""Write data/derived/_pot_grid.png: tray A with the pot grid, ROI, ROI+margin clip and per-pot areas drawn on it.

    python scripts/pot_grid_preview.py [--tray A] [--frame A:2016-01-06T09:00:00Z] [--db data/ceres.db]

Uses the pots_json stored on the frame's canopy row (run scripts/backfill_pots.py first); empty pots are grey.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from backend.config import load_settings, load_trays
from backend.db import connect
from backend.perception import pots


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(ROOT / load_settings().db_path))
    ap.add_argument("--tray", default="A")
    ap.add_argument("--frame", help="frame id (default: the first A frame of 2016-01-06)")
    ap.add_argument("--out", default=str(ROOT / "data" / "derived" / "_pot_grid.png"))
    a = ap.parse_args(argv)
    tray = load_trays().by_id(a.tray)
    conn = connect(a.db)
    q = "SELECT f.id, f.path, m.pots_json FROM frames f JOIN masks m ON m.frame_id = f.id AND m.kind = 'canopy'"
    row = (
        conn.execute(q + " WHERE f.id = ?", (a.frame,)).fetchone()
        if a.frame
        else conn.execute(q + " WHERE f.tray_id = ? AND f.ts >= '2016-01-06' ORDER BY f.ts LIMIT 1", (a.tray,)).fetchone()
    )
    img = cv2.imread(str(ROOT / row["path"]))
    area = json.loads(row["pots_json"] or "{}")
    cx, cy, cw, ch = pots.clip_region(tray, img.shape[:2])
    cv2.rectangle(img, (cx, cy), (cx + cw, cy + ch), (255, 200, 0), 1)  # ROI + margin: the canopy clip
    x, y, w, h = tray.roi
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)  # ROI
    for (r, c), (x0, y0, x1, y1) in pots.pot_bounds(tray).items():
        pid = pots.pot_id(tray.id, r, c)
        held = area.get(pid, 0) >= pots.POT_MIN_PX
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255) if held else (160, 160, 160), 2)
        cv2.putText(img, f"{r}{c}", (x0 + 6, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, str(area.get(pid, 0)), (x0 + 6, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"{row['id']}  plants: {pots.plant_count(area)}", (10, 760), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, img)
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
