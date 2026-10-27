"""Regenerate tests/golden/tray_b.json:  python tests/golden/gen_tray_b.py

Sensor data comes from the deterministic simulated hub (seed 0, 5-min steps): B:soil_vwc for the whole
window (the WILT baseline needs the 48 h before fault_start), the other series for the last 6 h only.
The canopy series are synthetic (B shrinks after fault_start and crashes on 2016-01-07 afternoon; A grows).
The expected alerts are what the rules return, reviewed by hand; rerunning rewrites them, so check the diff.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sensor_reader import ingest_lines

from backend.config import load_trays
from backend.db import init_db
from backend.science import rules
from backend.sim.sensor_hub import SimulatedHub
from backend.timeutil import iso_z, parse_ts

START, END, AS_OF = "2016-01-03T08:00:00Z", "2016-01-07T18:00:00Z", "2016-01-07T18:00:00Z"
TAIL_FROM = "2016-01-07T12:00:00Z"
FAULT_START = parse_ts("2016-01-05T08:00:00Z")


def canopy(tray: str, wilting: bool) -> list[dict]:
    rows, t = [], parse_ts(START)
    while t <= parse_ts(END):
        if 9 <= t.hour <= 21 and t.hour % 2 == 1 and t.minute == 0:  # imaging every 2 h, 09:00-21:00
            d = (t - parse_ts("2015-12-14T00:00:00Z")).total_seconds() / 86400
            area, exg = 8000 * math.exp(0.07 * d), 0.32
            if wilting and t > FAULT_START:
                days = (t - FAULT_START).total_seconds() / 86400
                crash = 1 / (1 + math.exp(-((t - parse_ts("2016-01-07T13:00:00Z")).total_seconds() / 3600)))
                area *= (1 - 0.06 * days) * (1 - 0.18 * crash)
                exg *= (1 - 0.02 * days) * (1 - 0.20 * crash)
            rows.append({"id": f"mask:{tray}:{t:%Y%m%dT%H%M%S}", "ts": iso_z(t), "area_px": round(area), "exg": round(exg, 4)})
        t += timedelta(hours=1)
    return rows


def main() -> None:
    cfg = load_trays()
    conn = init_db(":memory:")
    ingest_lines(conn, SimulatedHub(cfg, 0, 300, start=START, end=END).lines(), START, cfg)

    def series(owner: str, metric: str, since: str = START) -> list[dict]:
        q = "SELECT id, ts, value, source FROM sensor_readings WHERE tray_id=? AND metric=? AND ts>=? ORDER BY ts"
        return [dict(r) for r in conn.execute(q, (owner, metric, since))]

    readings = {"B:soil_vwc": series("B", "soil_vwc")}
    for owner, metric in [("B", "soil_raw"), ("B", "lamp_on"), ("air", "temp_c"), ("air", "rh_pct")]:
        readings[f"{owner}:{metric}"] = series(owner, metric, TAIL_FROM)
    c_b, c_a = canopy("B", True), canopy("A", False)
    rd = {tuple(k.split(":")): v for k, v in readings.items()}
    alerts = rules.evaluate(cfg.by_id("B"), rd, c_b, c_a, AS_OF)
    expected = [
        {"rule": a.rule, "severity": a.severity, "tray_id": a.tray_id, "ts": a.ts,
         "evidence_ids": list(a.evidence_ids), "threshold_tag": a.threshold_tag}
        for a in alerts
    ]
    out = {
        "_note": "Synthetic tray B window: hub seed 0 at 5-min steps, synthetic canopy. See gen_tray_b.py.",
        "tray_id": "B", "as_of": AS_OF, "readings": readings, "canopy": c_b, "ref_canopy": c_a,
        "expected_alerts": expected,
    }
    (Path(__file__).with_name("tray_b.json")).write_text(json.dumps(out, separators=(",", ":")))
    print(json.dumps(expected, indent=1))


if __name__ == "__main__":
    main()
