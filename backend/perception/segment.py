"""segment_frames (M2): segment frames that have no canopy mask yet and write masks rows with metrics.

Canopy is clipped to the tray ROI grown by `roi_margin_px` (clamped to the frame). Each plant instance is
assigned to the pot holding its centroid (`plants` rows, `masks.plant_id`) and the canopy row carries the
per-pot area in `pots_json` (backend/perception/pots.py).

The SAM cache is keyed on (raw frame sha, fault_hash, prompt, backend), see sam_client.cache_key.
`recompute_from_cache` rebuilds every masks row from the cache alone, with no model call.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from pathlib import Path

import cv2

from backend.config import ROOT, TrayConfig, TraysConfig
from backend.perception import metrics, pots
from backend.perception.sam_client import (
    CACHE_DIR,
    ApiBackend,
    HsvBackend,
    Mask,
    SamBackend,
    cache_path,
    legacy_cache_key,
    segment_cached,
)
from backend.spend import BudgetExceeded, cost_for

PROMPT = "plant"
RAW_DIR = ROOT / "data" / "raw" / "aled"
_COLS = "f.id, f.tray_id, f.ts, f.path, f.cm2_per_px, f.fault_hash"


class CacheMiss(RuntimeError):
    """recompute_from_cache found frames with no cached SAM payload."""


def pending_frames(
    conn: sqlite3.Connection, trays: list[str] | None = None, as_of: str | None = None
) -> list[sqlite3.Row]:
    q = (
        f"SELECT {_COLS} FROM frames f WHERE NOT EXISTS "
        "(SELECT 1 FROM masks m WHERE m.frame_id = f.id AND m.kind = 'canopy')"
    )
    args: list[str] = []
    if as_of:  # ISO-Z strings compare chronologically
        q += " AND f.ts <= ?"
        args.append(as_of)
    if trays:
        q += f" AND f.tray_id IN ({','.join('?' * len(trays))})"
        args += list(trays)
    return conn.execute(q + " ORDER BY f.ts, f.tray_id", args).fetchall()


def frame_source(frame: sqlite3.Row, raw_dir: Path | None = None) -> tuple[Path, Path | None, str | None]:
    """(image path, raw ALED frame, fault_hash) for cache keying. The raw frame is the same-named file in
    the dataset folder for a derived frame (fault_hash set); None for a dataset frame (the image is raw)."""
    image = ROOT / frame["path"]
    fh = frame["fault_hash"]
    raw = (raw_dir or RAW_DIR) / image.name
    return image, (raw if fh and raw.is_file() else None), fh


def frame_cache_path(
    frame: sqlite3.Row, backend_name: str = "api", cache_dir: Path = CACHE_DIR, prompt: str = PROMPT
) -> Path:
    image, raw, fh = frame_source(frame)
    return cache_path(image, prompt, backend_name, cache_dir, raw, fh)


def estimate_cost_usd(
    conn: sqlite3.Connection, frames: list[sqlite3.Row], cache_dir: Path = CACHE_DIR
) -> float:
    """$ for frames not already in the disk cache, at sam-3.1 prices."""
    n = sum(1 for f in frames if not frame_cache_path(f, "api", cache_dir).exists())
    return cost_for("sam-3.1", images=n)


def migrate_cache(
    conn: sqlite3.Connection, cache_dir: Path = CACHE_DIR, backends: tuple[str, ...] = ("api", "hsv")
) -> dict:
    """Copy cache entries stored under the legacy key (sha of the derived image bytes) to the new key
    (raw sha + fault_hash) for every derived frame, so nothing is billed twice. Idempotent; old files stay."""
    out = {"copied": 0, "already": 0, "missing": 0}
    for f in conn.execute(f"SELECT {_COLS} FROM frames f WHERE f.fault_hash IS NOT NULL").fetchall():
        image = ROOT / f["path"]
        for b in backends:
            new = frame_cache_path(f, b, cache_dir)
            old = cache_dir / f"{legacy_cache_key(image, PROMPT, b)}.json" if image.is_file() else None
            if new.exists():
                out["already"] += 1
            elif old is not None and old.exists():
                shutil.copyfile(old, new)
                out["copied"] += 1
            elif b == backends[0]:
                out["missing"] += 1
    return out


def write_masks(conn: sqlite3.Connection, frame: sqlite3.Row, masks: list[Mask], tray: TrayConfig) -> None:
    """Replace the frame's masks rows: canopy (union, clipped to ROI + margin, with pots_json) and one
    `plant` row per instance, assigned to its pot (masks.plant_id, plants rows)."""
    img = cv2.imread(str(ROOT / frame["path"]))
    shape = img.shape[:2]
    clip = metrics.roi_mask(shape, pots.clip_region(tray, shape))
    rasters = [m.raster() & clip for m in masks]
    canopy = metrics.union_mask(rasters, shape, None)
    pot_ids, pot_area = pots.assign(tray, rasters)
    a = metrics.area_px(canopy)
    fid, tid, cm2 = frame["id"], frame["tray_id"], frame["cm2_per_px"]
    conn.execute("DELETE FROM masks WHERE frame_id = ?", (fid,))
    conn.execute(
        "INSERT INTO masks(id, frame_id, tray_id, plant_id, kind, area_px, area_cm2, exg, chlorosis_frac, rle,"
        " pots_json) VALUES (?,?,?,NULL,'canopy',?,?,?,?,?,?)",
        (
            f"{fid}:canopy", fid, tid, a, metrics.area_cm2(a, cm2), metrics.exg(img, canopy),
            metrics.chlorosis_frac(img, canopy), json.dumps(metrics.rle_encode(canopy)), json.dumps(pot_area),
        ),
    )
    held = pots.holding(pot_area)
    for pid in held:
        conn.execute(
            "INSERT INTO plants(id, tray_id, first_seen_ts, label) VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE"
            " SET first_seen_ts = MIN(first_seen_ts, excluded.first_seen_ts)",
            (pid, tid, frame["ts"], f"pot {pid.rsplit('pot', 1)[1]}"),
        )
    for i, (r, pid) in enumerate(zip(rasters, pot_ids)):
        pa = metrics.area_px(r)
        if pa == 0:
            continue
        conn.execute(
            "INSERT INTO masks(id, frame_id, tray_id, plant_id, kind, area_px, area_cm2, exg, chlorosis_frac, rle)"
            " VALUES (?,?,?,?,'plant',?,?,?,?,?)",
            (
                f"{fid}:plant:{i}", fid, tid, pid if pid in held else None, pa, metrics.area_cm2(pa, cm2),
                metrics.exg(img, r), metrics.chlorosis_frac(img, r), json.dumps(metrics.rle_encode(r)),
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
    as_of: str | None = None,
) -> dict:
    """Segment frames lacking a canopy row. DB access stays on the event-loop thread; calls run concurrently."""
    frames = pending_frames(conn, trays, as_of)[: limit if limit is not None else None]
    stats = {"frames": len(frames), "segmented": 0, "cache_hits": 0, "errors": 0, "budget_blocked": 0}
    by_id = {t.id: t for t in cfg.trays}

    async def one(sem: asyncio.Semaphore, f: sqlite3.Row) -> None:
        async with sem:
            image, raw, fh = frame_source(f)
            try:
                masks, hit = await segment_cached(backend, image, PROMPT, cache_dir, raw, fh)
            except BudgetExceeded:
                stats["budget_blocked"] += 1
                return
            except Exception as e:  # noqa: BLE001  keep the batch going; the frame stays pending
                stats["errors"] += 1
                print(f"segment failed for {f['id']}: {type(e).__name__}: {e}")
                return
        write_masks(conn, f, masks, by_id[f["tray_id"]])
        stats["segmented"] += 1
        stats["cache_hits"] += int(hit)

    async def main() -> None:
        sem = asyncio.Semaphore(concurrency)
        await asyncio.gather(*(one(sem, f) for f in frames))

    asyncio.run(main())
    return stats


def recompute_from_cache(
    conn: sqlite3.Connection,
    cfg: TraysConfig,
    backend_name: str = "api",
    trays: list[str] | None = None,
    cache_dir: Path = CACHE_DIR,
) -> dict:
    """Rebuild the masks rows of every frame from .sam_cache alone (no model call, no spend). Refuses, writing
    nothing, with CacheMiss when any frame is not cached."""
    q = f"SELECT {_COLS} FROM frames f"
    args: list[str] = []
    if trays:
        q += f" WHERE f.tray_id IN ({','.join('?' * len(trays))})"
        args = list(trays)
    frames = conn.execute(q + " ORDER BY f.ts, f.tray_id", args).fetchall()
    paths = {f["id"]: frame_cache_path(f, backend_name, cache_dir) for f in frames}
    missing = [fid for fid, p in paths.items() if not p.exists()]
    if missing:
        raise CacheMiss(f"{len(missing)} of {len(frames)} frames are not cached, e.g. {missing[:3]}")
    backend = ApiBackend(conn) if backend_name == "api" else HsvBackend()  # decode only: no client is built
    by_id = {t.id: t for t in cfg.trays}

    async def main() -> None:
        for f in frames:
            payload = json.loads(paths[f["id"]].read_text(encoding="utf-8"))["payload"]
            masks = await backend.adecode(payload, ROOT / f["path"])
            write_masks(conn, f, masks, by_id[f["tray_id"]])

    asyncio.run(main())
    return {"frames": len(frames), "recomputed": len(frames)}
