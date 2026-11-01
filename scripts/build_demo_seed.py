"""Export the offline demo bundle from data/ceres.db (opened READ-ONLY, never written) into data/demo/.

    python scripts/build_demo_seed.py [--source-db data/ceres.db] [--width 640] [--quality 65]

Writes
  data/demo/seed.db.gz     compact SQLite (gzip): trays, frames, canopy masks (RLE at the demo width, pots_json kept),
                           plants, sensor_readings, alerts, alert_events. No spend, assessments, actions or queue rows.
  data/demo/frames/*.jpg   every frame the UI shows (raw and fault-derived, as the image endpoint serves them),
                           downscaled to --width px and re-encoded as JPEG. The overlay endpoint draws the canopy mask
                           (rescaled to the image) on these.
Downscaling changes no served metric: area_px, area_cm2, exg, chlorosis_frac and pots_json are copied from the DB.
ALED frames are CC BY 4.0 (see NOTICE): the bundle contains downscaled, re-encoded ALED frames (and, for trays B and C
after their fault start, the labelled simulated-fault transforms of them).
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.db import SCHEMA
from backend.perception.metrics import rle_decode, rle_encode

DEMO = ROOT / "data" / "demo"
FRAMES = DEMO / "frames"
COPY = ("trays", "plants", "sensor_readings", "alerts", "alert_events")


def frame_name(fid: str) -> str:
    tray, ts = fid.split(":", 1)
    return f"{tray}_{ts.replace('-', '').replace(':', '').rstrip('Z')}.jpg"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source-db", type=Path, default=ROOT / "data" / "ceres.db")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--quality", type=int, default=65)
    ap.add_argument("--out-dir", type=Path, default=DEMO, help="bundle directory (default data/demo; tests use a temp dir)")
    a = ap.parse_args(argv)
    demo, frames_dir = a.out_dir, a.out_dir / "frames"
    try:
        rel = demo.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        rel = demo.resolve().as_posix()
    src = sqlite3.connect(a.source_db.resolve().as_uri() + "?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    tmp = Path(tempfile.mkdtemp(prefix="ceres_seed_"))
    out = sqlite3.connect(tmp / "seed.db")
    out.executescript(SCHEMA)
    for t in COPY:
        cols = [r[1] for r in src.execute(f"PRAGMA table_info({t})")]
        q = ",".join("?" * len(cols))
        out.executemany(f"INSERT INTO {t}({','.join(cols)}) VALUES ({q})", (tuple(r) for r in src.execute(f"SELECT * FROM {t}")))
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in list(frames_dir.glob("*")):  # replaces the earlier 512 px subset (raw frames only) with the full bundle
        if old.is_file():
            old.unlink()
    n = 0
    fcols = [r[1] for r in src.execute("PRAGMA table_info(frames)")]
    for f in src.execute("SELECT * FROM frames ORDER BY ts, tray_id"):
        img = cv2.imread(str(ROOT / f["path"]))
        if img is None:
            print(f"error: cannot read {f['path']}", file=sys.stderr)
            return 1
        h, w = img.shape[:2]
        nw, nh = a.width, round(h * a.width / w)
        name = frame_name(f["id"])
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(frames_dir / name), small, [cv2.IMWRITE_JPEG_QUALITY, a.quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
        row = {c: f[c] for c in fcols}
        row.update(path=f"{rel}/frames/{name}", fault_hash=None, src_file=None)
        out.execute(f"INSERT INTO frames({','.join(fcols)}) VALUES ({','.join('?' * len(fcols))})", [row[c] for c in fcols])
        for m in src.execute("SELECT * FROM masks WHERE frame_id = ? AND kind = 'canopy'", (f["id"],)):
            mask = cv2.resize(rle_decode(json.loads(m["rle"])).astype("uint8"), (nw, nh), interpolation=cv2.INTER_NEAREST)
            d = dict(m)
            d["rle"] = json.dumps(rle_encode(mask.astype(bool)), separators=(",", ":"))
            mc = list(d)
            out.execute(f"INSERT INTO masks({','.join(mc)}) VALUES ({','.join('?' * len(mc))})", [d[c] for c in mc])
        n += 1
    out.commit()
    out.execute("VACUUM")
    out.close()
    src.close()
    with open(tmp / "seed.db", "rb") as fi, open(demo / "seed.db.gz", "wb") as raw,             gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=9) as gz:
        shutil.copyfileobj(fi, gz)
    shutil.rmtree(tmp, ignore_errors=True)
    total = sum(p.stat().st_size for p in demo.rglob("*") if p.is_file())
    imgs = sum(p.stat().st_size for p in frames_dir.glob("*.jpg"))
    print(f"{n} frames; images {imgs / 1e6:.1f} MB; seed {(demo / 'seed.db.gz').stat().st_size / 1e6:.1f} MB; "
          f"data/demo total {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
