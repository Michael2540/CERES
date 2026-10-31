"""Bias check for the SAM colour trim: does it remove more canopy from the faulted trays than from tray A?

python scripts/trim_bias.py --db data/ceres.backup-m6.db [--s-min 40] [--v-rel 0.7] [--close 5] [--step 3]

Reads the UNTRIMMED instance masks of a DB (the pre-trim backup), trims each frame's instances exactly as
segment.write_masks does, and reports per tray the median fraction of canopy the trim removes, before and
after the tray's fault start (tray B wilt transform, tray C brightness x0.6 time-warp). Tray A is reported
over the same dates as the faulted tray. A faulted tray losing ~5 points more than A would fake or hide
WILT_RISK / GROWTH_LAG, so the colour test must be loosened until it does not. No model call, no DB write.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from statistics import median

import cv2

from backend.config import ROOT, load_trays
from backend.perception import metrics, trim


def removed_fractions(conn: sqlite3.Connection, tray_id: str, p: tuple, close: int, step: int) -> list[tuple[str, float]]:
    """(frame ts, fraction of the untrimmed canopy the trim removes) for every `step`-th frame of a tray."""
    frames = conn.execute(
        "SELECT id, ts, path FROM frames WHERE tray_id = ? ORDER BY ts", (tray_id,)
    ).fetchall()[::step]
    out: list[tuple[str, float]] = []
    for f in frames:
        rows = conn.execute("SELECT rle FROM masks WHERE frame_id = ? AND kind = 'plant'", (f["id"],)).fetchall()
        img = cv2.imread(str(ROOT / f["path"]))
        if not rows or img is None:
            continue
        old = [metrics.rle_decode(json.loads(r["rle"])) for r in rows]
        new = trim.trim_instances(img, old, *p, close)
        a0 = metrics.area_px(metrics.union_mask(old, img.shape[:2], None))
        a1 = metrics.area_px(metrics.union_mask(new, img.shape[:2], None))
        out.append((f["ts"], 1.0 - a1 / a0 if a0 else 0.0))
    return out


def med(xs: list[float]) -> float | None:
    return round(100 * median(xs), 1) if xs else None


def report(conn: sqlite3.Connection, p: tuple, close: int, step: int) -> dict:
    cfg = load_trays()
    fr = {t.id: removed_fractions(conn, t.id, p, close, step) for t in cfg.trays if t.id in "ABC"}
    res: dict = {}
    for t in cfg.trays:
        if t.id not in fr or t.fault_start is None or t.id == "A":
            continue
        s = t.fault_start.replace("+00:00", "Z")
        res[t.id] = {
            "before": med([v for ts, v in fr[t.id] if ts < s]), "after": med([v for ts, v in fr[t.id] if ts >= s]),
            "A_before": med([v for ts, v in fr["A"] if ts < s]), "A_after": med([v for ts, v in fr["A"] if ts >= s]),
        }
        res[t.id]["extra_loss_pp"] = (
            None if res[t.id]["after"] is None or res[t.id]["A_after"] is None
            else round(res[t.id]["after"] - res[t.id]["A_after"], 1)
        )
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--s-min", type=int, default=trim.TRIM_S_MIN)
    ap.add_argument("--v-abs", type=int, default=trim.TRIM_V_ABS)
    ap.add_argument("--v-rel", type=float, default=trim.TRIM_V_REL)
    ap.add_argument("--close", type=int, default=trim.TRIM_CLOSE_PX)
    ap.add_argument("--step", type=int, default=1, help="use every n-th frame")
    a = ap.parse_args()
    conn = sqlite3.connect(a.db)
    conn.row_factory = sqlite3.Row
    r = report(conn, (a.s_min, a.v_abs, a.v_rel), a.close, a.step)
    print(f"trim s_min={a.s_min} v_abs={a.v_abs} v_rel={a.v_rel} close={a.close}: median % of canopy removed (A over the same dates as the faulted tray)")
    for tid, v in r.items():
        print(f"  tray {tid}: before {v['before']}  after {v['after']}  | A before {v['A_before']}  A after {v['A_after']}"
              f"  | extra loss vs A (after): {v['extra_loss_pp']} pp")


if __name__ == "__main__":
    main()
