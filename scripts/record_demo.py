"""Record live Flight Agronomist runs for DEMO_MODE (M9). PAID: one live muse-spark-1.3 assessment per alert (~$0.10-0.20).

    python scripts/record_demo.py --all-demo --dry-run                                  # plan + cost estimate, no calls
    python scripts/record_demo.py --alert <alert_id> --as-of 2016-01-07T12:00:00Z       # one alert
    python scripts/record_demo.py --all-demo                                             # the four demo alerts (PAID)

The four demo alerts are the same as the edge recordings (scripts/record_edge.py), so the Mars-mode drain audit replays too.
Each run happens in a temp copy of the source DB (default: DB_PATH, opened read-only; never written) with DEMO_MODE forced
off; spend goes to the MAIN ledger (budget guard applies). The recording (data/demo/agent_runs/<alert_id>.json) holds every
model response AND tool result plus the model name. A replay re-runs the citation and numeric-grounding checks; the stored
verdict is never trusted. An invalid run is NOT written unless --keep-invalid.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["DEMO_MODE"] = "0"  # a recording is always a live run (set before any settings are read)

from backend.agent import agronomist, replay
from backend.agent.llm_client import LLMClient
from backend.config import load_settings
from backend.db import connect
from backend.eval.run_eval import EST_PER_ASSESSMENT_USD, copy_db, reset_state
from backend.timeutil import parse_ts
from scripts.record_edge import DEMO, resolve_demo


def record_one(source: Path, ledger, alert_id: str, as_of: str, out_dir: Path, keep_invalid: bool) -> Path | None:
    work = Path(tempfile.mkdtemp(prefix="ceres_record_demo_"))
    conn = copy_db(source, work / "scenario.db")
    try:
        reset_state(conn)
        recorder = replay.RecordingLLM(LLMClient(ledger))
        aid = agronomist.assess(conn, alert_id, parse_ts(as_of), llm=recorder)
        rec = replay.export_recording(conn, aid, recorder)
    finally:
        conn.close()
        shutil.rmtree(work, ignore_errors=True)
    print(f"  {alert_id}: model {rec['model']}, valid {rec['valid']}, cost ${rec['cost_usd']:.4f}")
    if not rec["valid"]:
        print("    errors: " + "; ".join(rec["citation_errors"])[:400])
        if not keep_invalid:
            print("    NOT written (invalid); re-run, or pass --keep-invalid")
            return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = replay.recording_path(alert_id, out_dir)
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alert", help="alert id to record (needs --as-of)")
    ap.add_argument("--as-of", help="mission time of the assessment, ISO UTC")
    ap.add_argument("--all-demo", action="store_true", help="the four demo alerts: " + ", ".join(f"{t} {r} {a}" for t, r, a in DEMO))
    ap.add_argument("--source-db", type=Path, default=None, help="DB to replay from (default: DB_PATH); never written")
    ap.add_argument("--out-dir", type=Path, default=replay.RUNS_DIR)
    ap.add_argument("--keep-invalid", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the estimated cost; no model call")
    args = ap.parse_args(argv)
    if bool(args.alert) == args.all_demo or (args.alert and not args.as_of):
        print("error: give either --alert <id> --as-of <ts>, or --all-demo", file=sys.stderr)
        return 2
    settings = load_settings()
    source = args.source_db or Path(settings.db_path)
    if not Path(source).is_file():
        print(f"error: source DB {source} not found", file=sys.stderr)
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="ceres_record_demo_plan_"))
    ro = copy_db(Path(source), tmp / "plan.db")  # migrated COPY: the source DB is never opened for writing
    try:
        targets = [(resolve_demo(ro, t, r, a), a) for t, r, a in DEMO] if args.all_demo else [(args.alert, args.as_of)]
    except agronomist.AlertNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        ro.close()
        shutil.rmtree(tmp, ignore_errors=True)
    lo, hi = EST_PER_ASSESSMENT_USD
    print(f"{len(targets)} live Flight Agronomist run(s) with muse-spark-1.3; estimated ${len(targets) * lo:.2f} to "
          f"${len(targets) * hi:.2f} (${lo:.2f} to ${hi:.2f} each). Budget cap ${settings.budget_usd:.2f}.")
    for aid, ts in targets:
        print(f"  {aid} at {ts}")
    if args.dry_run:
        return 0
    ledger = connect(settings.db_path)
    try:
        written = [record_one(Path(source), ledger, a, ts, args.out_dir, args.keep_invalid) for a, ts in targets]
    finally:
        ledger.close()
    print(f"wrote {sum(1 for w in written if w)} of {len(targets)} recordings to {args.out_dir}")
    return 0 if all(written) else 1


if __name__ == "__main__":
    raise SystemExit(main())
