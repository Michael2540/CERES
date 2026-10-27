"""Fill the database with simulated sensor data for the whole dataset span (config/sim.yaml)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sensor_reader import simulate_into_db

from backend.config import load_trays
from backend.db import init_db
from backend.sim.sensor_hub import load_sim_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/ceres.db", help="SQLite path")
    ap.add_argument("--start", default="2015-12-14T00:00:00Z")
    ap.add_argument("--end", default="2016-01-18T23:59:00Z")
    ap.add_argument("--keep", action="store_true", help="keep existing simulated rows (default: replace them)")
    args = ap.parse_args(argv)
    conn = init_db(args.db)
    n = simulate_into_db(conn, load_trays(), load_sim_config(), args.start, args.end, replace=not args.keep)
    print(f"{n} rows inserted")
    for metric, tray, count in conn.execute(
        "SELECT metric, tray_id, COUNT(*) FROM sensor_readings GROUP BY metric, tray_id ORDER BY metric, tray_id"
    ):
        print(f"  {metric:9s} {tray:4s} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
