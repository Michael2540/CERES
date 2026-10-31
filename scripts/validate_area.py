"""SAM 3.1 vs ALED hand-annotated ground truth (raw frames in data/raw/aled_gt).

python scripts/validate_area.py --dry-run        # count cached vs uncached + estimated cost, no calls
python scripts/validate_area.py --cached-only    # free: only frames already in .sam_cache (marks partial)
python scripts/validate_area.py                  # paid for uncached frames (prints the cost first), writes the JSON
python scripts/validate_area.py --backend hsv    # free colour-threshold baseline only, prints, writes nothing

Paid output goes into the same .sam_cache (never paid twice) and through the spend ledger and budget guard.
Writes backend/eval/out/area_validation.json and overlays of the 3 worst frames to data/derived/validation/.
It reads the SAM cache only: no frames or masks rows are written to the mission tables.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from backend.config import ROOT, load_settings, load_trays
from backend.db import init_db
from backend.eval import validate_area as va
from backend.perception.sam_client import ApiBackend, HsvBackend, cache_path, segment_cached
from backend.spend import BudgetExceeded, check_budget, cost_for, total_spend


def classify(pairs: list[tuple[str, Path, Path]]) -> tuple[list[str], list[str]]:
    """(cached stems, uncached stems) for the api backend, by cache lookup (dataset key: raw sha, no fault_hash)."""
    c = [s for s, j, _ in pairs if cache_path(j, va.PROMPT, "api").exists()]
    return c, [s for s, _, _ in pairs if s not in set(c)]


def hsv_rasters(_stem: str, jpg: Path) -> list[np.ndarray]:
    return [m.raster() for m in HsvBackend().segment(jpg, va.PROMPT)]


async def fetch_sam(
    pairs: list[tuple[str, Path, Path]], backend: ApiBackend, concurrency: int = 4
) -> tuple[dict[str, list[np.ndarray]], int, int]:
    """stem -> instance rasters through segment_cached (a hit makes no call), plus (cache hits, paid calls).
    A failed or budget-blocked frame is left out and reported."""
    out: dict[str, list[np.ndarray]] = {}
    stats = {"hit": 0, "called": 0}
    sem = asyncio.Semaphore(concurrency)

    async def one(stem: str, jpg: Path) -> None:
        async with sem:
            try:
                masks, hit = await segment_cached(backend, jpg, va.PROMPT)
            except BudgetExceeded as e:
                print(f"budget blocked {stem}: {e}")
                return
            except Exception as e:  # noqa: BLE001  keep the batch going; the frame is left out
                print(f"segment failed for {stem}: {type(e).__name__}: {e}")
                return
        out[stem] = [m.raster() for m in masks]
        stats["hit" if hit else "called"] += 1

    await asyncio.gather(*(one(s, j) for s, j, _ in pairs))
    return out, stats["hit"], stats["called"]


def decode_cached(pairs: list[tuple[str, Path, Path]]) -> dict[str, list[np.ndarray]]:
    """stem -> rasters from the cache alone (decode only: no client, no call, no spend)."""
    backend = ApiBackend(None)  # type: ignore[arg-type]  adecode never touches the connection

    async def main() -> dict[str, list[np.ndarray]]:
        out = {}
        for stem, jpg, _ in pairs:
            payload = json.loads(cache_path(jpg, va.PROMPT, "api").read_text(encoding="utf-8"))["payload"]
            out[stem] = [m.raster() for m in await backend.adecode(payload, jpg)]
        return out

    return asyncio.run(main())


def summary(title: str, agg: dict) -> None:
    o = agg["overall"]
    sm = agg["by_stage"]["small"]
    print(f"{title}: small-stage Dice={sm.get('dice_mean')} MAPE={sm.get('area_mape')};"
          f" instances: pot r={o.get('inst_pot_pearson_r')} plants exact={o.get('inst_plant_exact_frac')}"
          f" MAE={o.get('inst_plant_mae')}")
    print(f"{title}: n={o['n']} Dice mean={o.get('dice_mean')} median={o.get('dice_median')} min={o.get('dice_min')}"
          f" IoU mean={o.get('iou_mean')} pooled Dice={o.get('pooled_dice')} area r={o.get('area_pearson_r')}"
          f" MAPE={o.get('area_mape')} pot r={o.get('pot_pearson_r')} (n={o.get('pot_n')})"
          f" plants exact={o.get('plant_exact_frac')} MAE={o.get('plant_mae')}")
    for t, a in agg["by_tray"].items():
        print(f"   tray {t}: n={a['n']} Dice mean={a.get('dice_mean')} area r={a.get('area_pearson_r')}")
    for s in ("small", "large"):
        a = agg["by_stage"][s]
        print(f"   {s}: n={a['n']} Dice mean={a.get('dice_mean')} area r={a.get('area_pearson_r')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default=str(va.GT_DIR))
    ap.add_argument("--backend", choices=["api", "hsv"], default="api")
    ap.add_argument("--dry-run", action="store_true", help="count cached vs uncached + cost; no calls, no files")
    ap.add_argument("--cached-only", action="store_true", help="only frames already in the SAM cache (free); partial")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", help=f"JSON path (default {va.OUT_PATH.relative_to(ROOT)})")
    a = ap.parse_args()

    pairs = va.gt_pairs(Path(a.gt_dir))
    cfg = load_trays()
    cached, uncached = classify(pairs)
    est = cost_for("sam-3.1", images=len(uncached))
    print(f"{len(pairs)} GT pairs: {len(cached)} cached, {len(uncached)} uncached; estimated cost ${est:.4f}")
    if a.dry_run:
        return

    if a.backend == "hsv":
        rows = va.evaluate(pairs, cfg, hsv_rasters)
        summary("colour-threshold baseline, not a model", va.breakdowns(rows))
        return

    n_called, cost, conn = 0, 0.0, None
    if a.cached_only:
        use = [p for p in pairs if p[0] in set(cached)]
        sam = decode_cached(use)
        n_cached = len(use)
    else:
        load_settings()
        conn = init_db(ROOT / load_settings().db_path)  # the spend ledger only; no mission rows are written
        check_budget(conn, est)
        print(f"PAID: {len(uncached)} sam-3.1 calls, ${est:.4f}; ledger total ${total_spend(conn):.4f}")
        before = total_spend(conn)
        sam, n_cached, n_called = asyncio.run(fetch_sam(pairs, ApiBackend(conn), a.concurrency))
        cost = total_spend(conn) - before
        print(f"done: {n_called} calls, cost ${cost:.4f}, ledger total ${total_spend(conn):.4f}")

    rows = va.evaluate(pairs, cfg, lambda s, _j: sam.get(s), trimmed=True)
    raw_rows = va.evaluate(pairs, cfg, lambda s, _j: sam.get(s))  # the same masks without the colour trim
    base_rows = va.evaluate([p for p in pairs if p[0] in sam], cfg, hsv_rasters)
    by_id = {t.id: t for t in cfg.trays}
    ovs: dict[str, str] = {}
    for r in va.worst(rows):  # overlays (TP green, FP red, FN blue) of the 3 worst frames
        _, jpg, png = next(p for p in pairs if p[0] == r["stem"])
        tray = by_id[r["tray"]]
        gt = va.load_gt(png)
        canopy, _ = va.sam_canopy(sam[r["stem"]], tray, gt.shape, cv2.imread(str(jpg)))
        dst = va.OVERLAY_DIR / f"sam_{r['stem']}.png"
        va.write_overlay(dst, jpg, gt & va.clip_mask(tray, gt.shape), canopy)
        ovs[r["stem"]] = (dst.relative_to(ROOT) if dst.is_relative_to(ROOT) else dst).as_posix()
    partial = a.cached_only or len(rows) < len(pairs)
    report = va.build_report(
        rows, base_rows, raw_rows, n_gt=len(pairs), n_cached=n_cached, n_called=n_called, cost_usd=cost, partial=partial,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), worst_overlays=ovs,
    )
    summary("SAM 3.1 (colour-trimmed, production)", report["sam"])
    summary("SAM 3.1 raw masks (untrimmed)", report["sam_untrimmed"])
    summary("colour-threshold baseline, not a model", report["baseline_hsv"])
    for w in report["worst"]:
        print(f"worst: {w['frame']} Dice={w['dice']} gt={w['gt_px']} sam={w['sam_px']} overlay={w['overlay']}")
    out = Path(a.out) if a.out else va.OUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out} (partial={partial})")


if __name__ == "__main__":
    main()
