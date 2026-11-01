"""Build data/demo.db (gitignored) from the committed demo bundle data/demo/seed.db.gz. No network, no key, no cost.

    python scripts/seed_demo.py [--out data/demo.db]

The DB is rebuilt from scratch every time (an earlier demo.db, with its assessments and link state, is replaced).
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.db import init_db

SEED = ROOT / "data" / "demo" / "seed.db.gz"


def seed(out: Path, seed_gz: Path = SEED) -> Path:
    if not seed_gz.is_file():
        raise FileNotFoundError(f"{seed_gz} not found: the demo bundle is incomplete (scripts/build_demo_seed.py builds it)")
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-journal", "-wal", "-shm"):
        Path(str(out) + suffix).unlink(missing_ok=True)
    with gzip.open(seed_gz, "rb") as fi, open(out, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    init_db(out).close()  # idempotent schema + migrations
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "demo.db")
    a = ap.parse_args(argv)
    try:
        p = seed(a.out)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"seeded {p} ({p.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
