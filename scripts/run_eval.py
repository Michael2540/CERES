"""Agent eval harness. See backend/eval/run_eval.py.

    python scripts/run_eval.py --dry-run                        # scenario count + estimated cost, no calls
    python scripts/run_eval.py --role agronomist --effort low   # PAID (ledgered in data/ceres.db)
    python scripts/run_eval.py --fake --source-db some.db       # free scripted LLM
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.eval.run_eval import main

if __name__ == "__main__":
    raise SystemExit(main())
