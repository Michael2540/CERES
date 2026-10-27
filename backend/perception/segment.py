"""segment_frames (M2): segment frames that have no canopy mask yet and write masks rows with metrics."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import cv2

from backend.config import ROOT, TraysConfig
from backend.perception import metrics
from backend.perception.sam_client import CACHE_DIR, Mask, SamBackend, segment_cached
from backend.spend import BudgetExceeded, cost_for

PROMPT = "plant"


def pending_frames(conn: sqlite3.Connection, trays: list[str] | None = None) -> list[sqlite3.Row]:
    q = (
        "SELECT f.id, f.tray_id, f.ts, f.path, f.cm2_per_px FROM frames f WHERE NOT EXISTS "
        "(SELECT 1 FROM masks m WHERE m.frame_id = f.id AND m.kind = 'canopy')"
    )
    args: list[str] = []
    if trays:
        q += f" AND f.tray_id IN ({','.join('?' * len(trays))})"
        args = list(trays)
    return conn.execute(q + " ORDER BY f.ts, f.tray_id", args).fetchall()


def estimate_cost_usd(
    conn: sqlite3.Connection, frames: list[sqlite3.Row], cache_dir: Path = CACHE_DIR
) -> float:
    """$ for frames not already in the disk cache, at sam-3.1 prices."""
    from backend.perception.sam_client import cache_path

    n = sum(1 for f in frames if not cache_path(ROOT / f["path"], PROMPT, "api", cache_dir).exists())
    return cost_for("sam-3.1", images=n)


def write_masks(
    conn: sqlite3.Connection, frame: sqlite3.Row, masks: list[Mask], roi: tuple[int, int, int, int]
) -> None:
    img = cv2.imread(str(ROOT / frame["path"]))
    shape = img.shape[:2]
    rasters = [m.raster() for m in masks]
    canopy = metrics.union_mask(rasters, shape, roi)
    a = metrics.area_px(canopy)
    fid, tid = frame["id"], frame["tray_id"]
    conn.execute("DELETE FROM masks WHERE frame_id = ?", (fid,))
    conn.execute(
        "INSERT INTO masks(id, frame_id, tray_id, plant_id, kind, area_px, area_cm2, exg, chlorosis_frac, rle)"
        " VALUES (?,?,?,NULL,'canopy',?,?,?,?,?)",
        (
            f"{fid}:canopy",
            fid,
            tid,
            a,
            metrics.area_cm2(a, frame["cm2_per_px"]),
            metrics.exg(img, canopy),
            metrics.chlorosis_frac(img, canopy),
            json.dumps(metrics.rle_encode(canopy)),
        ),
    )
    for i, (m, r) in enumerate(zip(masks, rasters)):
        r = r & metrics.roi_mask(shape, roi)
        pa = metrics.area_px(r)
        if pa == 0:
            continue
        conn.execute(
            "INSERT INTO masks(id, frame_id, tray_id, plant_id, kind, area_px, area_cm2, exg, chlorosis_frac, rle)"
            " VALUES (?,?,?,NULL,'plant',?,?,?,?,?)",
            (
                f"{fid}:plant:{i}",
                fid,
                tid,
                pa,
                metrics.area_cm2(pa, frame["cm2_per_px"]),
                metrics.exg(img, r),
                metrics.chlorosis_frac(img, r),
                json.dumps(metrics.rle_encode(r)),
            ),
        )
    conn.commit()


def segment_frames(
    conn: sqlite3.Connection,
    cfg: TraysConfig,
    backend: SamBackend,
    limit: int | None = None,
    concurrency: int = 4,
    trays: list[str] | None = None,
    cache_dir: Path = CACHE_DIR,
) -> dict:
    """Segment frames lacking a canopy row. DB access stays on the event-loop thread; calls run concurrently."""
    frames = pending_frames(conn, trays)[: limit if limit is not None else None]
    stats = {"frames": len(frames), "segmented": 0, "cache_hits": 0, "errors": 0, "budget_blocked": 0}
    roi = {t.id: tuple(t.roi) for t in cfg.trays}

    async def one(sem: asyncio.Semaphore, f: sqlite3.Row) -> None:
        async with sem:
            try:
                masks, hit = await segment_cached(backend, ROOT / f["path"], PROMPT, cache_dir)
            except BudgetExceeded:
                stats["budget_blocked"] += 1
                return
            except Exception as e:  # noqa: BLE001  keep the batch going; the frame stays pending
                stats["errors"] += 1
                print(f"segment failed for {f['id']}: {type(e).__name__}: {e}")
                return
        write_masks(conn, f, masks, roi[f["tray_id"]])
        stats["segmented"] += 1
        stats["cache_hits"] += int(hit)

    async def main() -> None:
        sem = asyncio.Semaphore(concurrency)
        await asyncio.gather(*(one(sem, f) for f in frames))

    asyncio.run(main())
    return stats
