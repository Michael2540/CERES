"""Ingest data/raw/aled then segment frames lacking a canopy mask.

python scripts/segment_all.py --backend hsv --limit 3
python scripts/segment_all.py --dry-run          # frames to segment + estimated sam-3.1 cost, no calls
"""

from __future__ import annotations

import argparse

from backend.config import ROOT, load_settings, load_trays
from backend.db import init_db
from backend.perception.ingest import ingest_folder
from backend.perception.sam_client import get_backend
from backend.perception.segment import estimate_cost_usd, pending_frames, segment_frames
from backend.spend import total_spend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--trays", help="comma-separated tray ids, e.g. A,B")
    ap.add_argument("--backend", choices=["api", "hsv"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()

    settings, cfg = load_settings(), load_trays()
    conn = init_db(ROOT / settings.db_path)
    n = ingest_folder(conn, ROOT / "data" / "raw" / "aled", cfg)
    trays = a.trays.split(",") if a.trays else None
    pending = pending_frames(conn, trays)
    if a.limit is not None:
        pending = pending[: a.limit]
    print(f"ingested {n} new frames; {len(pending)} frames to segment")
    if a.dry_run:
        print(f"estimated sam-3.1 cost (uncached frames): ${estimate_cost_usd(conn, pending):.4f}")
        return
    backend = get_backend(a.backend, conn)
    stats = segment_frames(conn, cfg, backend, a.limit, a.concurrency, trays)
    print(f"backend={backend.name} model={backend.model} stats={stats} ledger_total=${total_spend(conn):.4f}")


if __name__ == "__main__":
    main()
