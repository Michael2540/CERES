"""Record live Edge Sentinel runs for EDGE_MODE=recorded (M8). PAID: one live muse-spark-1.3-contributor triage per alert.

    python scripts/record_edge.py --all-demo --dry-run                                   # plan + cost estimate, no calls
    python scripts/record_edge.py --alert <alert_id> --as-of 2016-01-07T12:00:00Z        # one alert
    python scripts/record_edge.py --all-demo                                              # the four demo alerts

Each run happens in a temp copy of the source DB (default: DB_PATH, opened read-only; it is never written) with
EDGE_MODE forced to spark_contributor; spend goes to the MAIN ledger (budget guard applies). The recording
(data/demo/edge_runs/<alert_id>.json) holds the model responses AND the tool results, plus the model name, so a replay is
free, offline and writes the same assessments / tool_log rows. A recording whose triage was invalid (schema, citation or
grounding errors) is NOT written unless --keep-invalid.
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

from backend.agent.agronomist import AlertNotFound
from backend.agent.llm_client import LLMClient
from backend.api.alerts import visible_alerts
from backend.config import load_settings
from backend.db import connect
from backend.eval.run_eval import EST_PER_EDGE_USD, copy_db, reset_state
from backend.link import edge
from backend.link.delay_sim import LinkState
from backend.timeutil import parse_ts

# (tray, rule, as_of): the alert open at that time on that tray (the latest episode when several are visible)
DEMO = [
    ("C", "GROWTH_LAG", "2016-01-07T12:00:00Z"),
    ("B", "WILT_RISK", "2016-01-07T12:00:00Z"),
    ("L", "LOW_VPD_RISK", "2016-01-09T12:00:00Z"),
    ("air", "SENSOR_FAULT", "2016-01-10T15:00:00Z"),
]


def resolve_demo(conn, tray: str, rule: str, as_of: str) -> str:
    cands = [a for a in visible_alerts(conn, parse_ts(as_of), tray) if a["rule"] == rule and a["status"] == "open"]
    if not cands:
        raise AlertNotFound(f"no open {rule} alert on tray {tray} at {as_of}")
    return max(cands, key=lambda a: a["ts"])["id"]


def record_one(source: Path, ledger, alert_id: str, as_of: str, out_dir: Path, keep_invalid: bool) -> Path | None:
    work = Path(tempfile.mkdtemp(prefix="ceres_record_edge_"))
    conn = copy_db(source, work / "scenario.db")
    try:
        reset_state(conn)
        recorder = edge.RecordingLLM(LLMClient(ledger))
        aid = edge.triage(conn, alert_id, parse_ts(as_of), llm=recorder, link=LinkState("down"))
        rec = edge.export_recording(conn, aid, recorder, os.environ["EDGE_MODE"])
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
    path = edge.recording_path(alert_id, out_dir)
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alert", help="alert id to record (needs --as-of)")
    ap.add_argument("--as-of", help="mission time of the triage, ISO UTC")
    ap.add_argument("--all-demo", action="store_true", help="the four demo alerts: " + ", ".join(f"{t} {r} {a}" for t, r, a in DEMO))
    ap.add_argument("--source-db", type=Path, default=None, help="DB to replay from (default: DB_PATH); never written")
    ap.add_argument("--out-dir", type=Path, default=edge.RUNS_DIR)
    ap.add_argument("--keep-invalid", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the estimated cost; no model call")
    args = ap.parse_args(argv)
    if bool(args.alert) == args.all_demo or (args.alert and not args.as_of):
        print("error: give either --alert <id> --as-of <ts>, or --all-demo", file=sys.stderr)
        return 2
    os.environ["EDGE_MODE"] = "spark_contributor"  # a recording is always a live contributor run
    settings = load_settings()
    source = args.source_db or Path(settings.db_path)
    if not Path(source).is_file():
        print(f"error: source DB {source} not found", file=sys.stderr)
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="ceres_record_edge_plan_"))
    ro = copy_db(Path(source), tmp / "plan.db")  # migrated COPY: the source DB is never opened for writing
    try:
        if args.all_demo:
            targets = [(resolve_demo(ro, t, r, a), a) for t, r, a in DEMO]
        else:
            targets = [(args.alert, args.as_of)]
    except AlertNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        ro.close()
        shutil.rmtree(tmp, ignore_errors=True)
    lo, hi = EST_PER_EDGE_USD
    print(f"{len(targets)} live edge triage(s) with muse-spark-1.3-contributor; estimated ${len(targets) * lo:.2f} to "
          f"${len(targets) * hi:.2f} (${lo:.3f} to ${hi:.3f} each). Budget cap ${settings.budget_usd:.2f}.")
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
