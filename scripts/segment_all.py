"""Ingest data/raw/aled then segment frames lacking a canopy mask.

python scripts/segment_all.py --backend hsv --limit 3
python scripts/segment_all.py --dry-run          # frames to segment + estimated sam-3.1 cost, no calls
python scripts/segment_all.py --recompute-from-cache   # rebuild masks rows (ROI clip, pots) from .sam_cache only

Cache entries stored under the pre-fault_hash key are copied to the current key first (no re-billing).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.config import ROOT, load_settings, load_trays
from backend.db import init_db
from backend.perception.ingest import DERIVED_DIR, ingest_folder
from backend.perception.sam_client import get_backend
from backend.perception.segment import (
    CacheMiss,
    estimate_cost_usd,
    migrate_cache,
    pending_frames,
    recompute_from_cache,
    segment_frames,
)
from backend.spend import total_spend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--trays", help="comma-separated tray ids, e.g. A,B")
    ap.add_argument("--backend", choices=["api", "hsv"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument(
        "--recompute-from-cache",
        action="store_true",
        help="rebuild masks rows from .sam_cache only (no calls); refuses if any frame is uncached",
    )
    ap.add_argument("--db", help="database path (default: DB_PATH from the environment / data/ceres.db)")
    ap.add_argument("--derived-dir", help="where regenerated fault frames are written (default: data/derived)")
    a = ap.parse_args()

    settings, cfg = load_settings(), load_trays()
    conn = init_db(a.db or ROOT / settings.db_path)
    print(f"cache migration: {migrate_cache(conn)}")  # before ingest, while derived bytes are still the old ones
    if a.recompute_from_cache:
        try:
            stats = recompute_from_cache(conn, cfg, a.backend or "api", a.trays.split(",") if a.trays else None)
        except CacheMiss as e:
            raise SystemExit(f"refusing: {e}") from None
        print(f"recomputed masks from cache: {stats}")
        return
    n = ingest_folder(conn, ROOT / "data" / "raw" / "aled", cfg, Path(a.derived_dir) if a.derived_dir else DERIVED_DIR)
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
