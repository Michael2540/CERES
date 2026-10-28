"""Build or refresh a DB for the demo timeline, then replay the pipeline hour by hour.

    python scripts/replay.py --db data/demo.db --backend none --from 2015-12-16T00:00:00Z --to 2016-01-18T22:00:00Z

Steps: ingest frames, simulate sensors (skipped when readings exist, unless --resimulate), optionally
segment frames (--backend hsv|api; none = use the masks already in the DB), then run_cycle at every hour
from --from to --to so alerts accumulate exactly as they would live. The alerts table is cleared first
(--keep-alerts to keep it), so a replay is deterministic. Prints a per-day alert summary.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sensor_reader import simulate_into_db

from backend.config import load_trays
from backend.db import init_db
from backend.perception.ingest import ingest_folder
from backend.perception.sam_client import get_backend
from backend.perception.segment import segment_frames
from backend.pipeline import RAW_DIR, run_cycle
from backend.sim.sensor_hub import load_sim_config
from backend.timeutil import iso_z, parse_ts


def day_table(conn, start: str, end: str) -> list[str]:
    """One line per day: alerts first raised that day (marked +) and alerts still active (ts that day)."""
    new: dict[str, list[str]] = defaultdict(list)
    active: dict[str, list[str]] = defaultdict(list)
    for r in conn.execute("SELECT id, first_ts, last_ts AS ts, tray_id, rule, detail FROM alerts ORDER BY first_ts"):
        first = r["first_ts"][:10]
        label = f"{r['tray_id']}:{r['rule']}"
        new[first].append(label)
        d = parse_ts(first + "T00:00:00Z")
        while iso_z(d)[:10] <= r["ts"][:10]:
            if iso_z(d)[:10] != first:
                active[iso_z(d)[:10]].append(label)
            d += timedelta(days=1)
    lines = [f"{'day':10s} | new | alerts first raised (+) / continuing"]
    d = parse_ts(start[:10] + "T00:00:00Z")
    while iso_z(d)[:10] <= end[:10]:
        day = iso_z(d)[:10]
        cells = [f"+{x}" for x in sorted(new[day])] + sorted(active[day])
        lines.append(f"{day:10s} | {len(new[day]):3d} | {', '.join(cells) if cells else '-'}")
        d += timedelta(days=1)
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/ceres.db")
    ap.add_argument("--from", dest="start", default="2015-12-16T00:00:00Z")
    ap.add_argument("--to", dest="end", default="2016-01-18T22:00:00Z")
    ap.add_argument("--backend", choices=["hsv", "api", "none"], default="none")
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--resimulate", action="store_true", help="re-run the simulated hub even if readings exist")
    ap.add_argument("--keep-alerts", action="store_true")
    ap.add_argument("--step-h", type=int, default=1)
    a = ap.parse_args(argv)

    cfg = load_trays()
    conn = init_db(a.db)
    t0 = time.time()
    n = ingest_folder(conn, a.raw_dir, cfg) if Path(a.raw_dir).is_dir() else 0
    print(f"ingested {n} new frames")
    have = conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]
    if a.resimulate or not have:
        rows = simulate_into_db(conn, cfg, load_sim_config(), replace=a.resimulate)
        print(f"simulated sensors: {rows} rows")
    if a.backend != "none":
        stats = segment_frames(conn, cfg, get_backend(a.backend, conn), as_of=a.end)
        print(f"segmentation ({a.backend}): {stats}")
    canopy = conn.execute("SELECT COUNT(*) FROM masks WHERE kind = 'canopy'").fetchone()[0]
    print(f"canopy masks in DB: {canopy}")
    if not a.keep_alerts:
        conn.execute("DELETE FROM alerts")
        conn.execute("DELETE FROM alert_events")
        conn.commit()

    t, end, cycles = parse_ts(a.start), parse_ts(a.end), 0
    while t <= end:
        run_cycle(conn, cfg, t, backend=None, segment=False, ingest=False)
        t += timedelta(hours=a.step_h)
        cycles += 1
    print(f"{cycles} cycles in {time.time() - t0:.1f}s\n")
    print("\n".join(day_table(conn, a.start, a.end)))
    print("\nepisodes (first raised -> last fired):")
    for r in conn.execute("SELECT tray_id, rule, first_ts, last_ts, detail,"
        " (SELECT COUNT(*) FROM alert_events e WHERE e.alert_id = a.id) AS n FROM alerts a ORDER BY first_ts"):
        print(f"  {r['tray_id']:3s} {r['rule']:13s} {r['first_ts']} -> {r['last_ts']} ({r['n']} events) {r['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
