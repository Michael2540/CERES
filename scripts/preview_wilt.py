"""Write data/derived/_preview_wilt.png: the same pot region of tray B at days 0, 1, 2, 4, 8 after fault_start."""

from __future__ import annotations

from datetime import UTC, datetime

import cv2
import numpy as np

from backend.config import ROOT, load_trays
from backend.sim.faults import wilt_transform

DAYS = (0, 1, 2, 4, 8)
REGION = (slice(10, 400), slice(60, 620))  # rows, cols: two rows of three pots


def main() -> None:
    cfg = load_trays().by_id("B")
    start = datetime.fromisoformat(cfg.fault_start)
    frames = {
        datetime.strptime(f.stem[2:], "%Y%m%dT%H%M%S").replace(tzinfo=UTC): f
        for f in (ROOT / "data" / "raw" / "aled").glob("B_*.jpg")
        if "_mask" not in f.stem
    }
    ref = cv2.imread(str(frames[max(t for t in frames if t <= start)]))
    tiles = []
    for d in DAYS:
        t = min(frames, key=lambda x: abs((x - start).total_seconds() / 86400 - d))
        days = (t - start).total_seconds() / 86400 if d else 0.0
        img = wilt_transform(cv2.imread(str(frames[t])), days, cfg.fault, tuple(cfg.roi), ref)
        tile = img[REGION].copy()
        cv2.putText(tile, f"day {d}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        tiles.append(tile)
    out = ROOT / "data" / "derived" / "_preview_wilt.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack(tiles))
    print(out)


if __name__ == "__main__":
    main()
