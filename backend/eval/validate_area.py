"""Area validation (M6): SAM 3.1 canopy masks against ALED hand-annotated ground truth on RAW frames.

`data/raw/aled_gt/` holds `<T>_<YYYYMMDDTHHMMSS>.jpg` (raw frame) and `<same>_gt.png` (RGBA label image;
background (0,0,0), every non-black pixel is plant). The SAM canopy is built through the production path
(cached raw output -> Mask rasters -> clip to ROI + margin -> union, segment.write_masks without the DB), and
everything is compared inside the same clip region.

Per pot, two ways. The GT label colours encode leaves, not plants, and touching rosettes merge into one
component, so the GT is always split by pot cell (each pixel to the pot `pots.pot_of` gives it).
- pixel split (`pot_*`, `plant_*`): the prediction's union is split the same way, so this measures mask quality
  per pot and is fair to a method without instances (the HSV baseline).
- instances (`inst_pot_*`, `inst_plant_*`): the prediction's instances go through `pots.assign` (centroid rule),
  as production does. A method that merges touching plants into one instance loses here.
Pooled per-pot r spans every growth stage (small and large pots), so read it with the by_stage numbers.

The HSV baseline is a colour threshold, not a model. Nothing here writes to the mission DB.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from backend.config import ROOT, TrayConfig, TraysConfig
from backend.perception import metrics, pots
from backend.perception.sam_client import SAM_MODEL

GT_DIR = ROOT / "data" / "raw" / "aled_gt"
OUT_PATH = ROOT / "backend" / "eval" / "out" / "area_validation.json"
OVERLAY_DIR = ROOT / "data" / "derived" / "validation"
PROMPT = "plant"
CAPTION = (
    "ALED hand-annotated ground truth (Bell & Dee 2016, CC BY 4.0) on real, untransformed frames of trays 031 and 032"
)
BASELINE_LABEL = "colour-threshold baseline, not a model"
_STEM = re.compile(r"^([A-Z])_(\d{8}T\d{6})$")


def frame_id(stem: str) -> str:
    """`A_20151214T125406` -> `A:2015-12-14T12:54:06Z` (the DB frame id convention)."""
    m = _STEM.match(stem)
    if m is None:
        raise ValueError(f"unexpected frame name {stem!r}")
    return f"{m[1]}:{datetime.strptime(m[2], '%Y%m%dT%H%M%S').strftime('%Y-%m-%dT%H:%M:%SZ')}"


def gt_pairs(gt_dir: Path = GT_DIR) -> list[tuple[str, Path, Path]]:
    """(stem, jpg, gt png) for every frame that has both files, sorted by name."""
    return [
        (j.stem, j, j.with_name(f"{j.stem}_gt.png"))
        for j in sorted(gt_dir.glob("*.jpg"))
        if _STEM.match(j.stem) and j.with_name(f"{j.stem}_gt.png").is_file()
    ]


def load_gt(path: str | Path) -> np.ndarray:
    """Boolean plant mask of a GT png: every pixel whose RGB is not black (alpha is ignored)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        return img > 0
    return img[..., :3].astype(np.int32).sum(axis=2) > 0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """2|A and B| / (|A| + |B|); 1.0 when both masks are empty."""
    a, b = a.astype(bool), b.astype(bool)
    tot = int(a.sum()) + int(b.sum())
    return 1.0 if tot == 0 else 2.0 * int(np.count_nonzero(a & b)) / tot


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """|A and B| / |A or B|; 1.0 when both masks are empty."""
    a, b = a.astype(bool), b.astype(bool)
    union = int(np.count_nonzero(a | b))
    return 1.0 if union == 0 else int(np.count_nonzero(a & b)) / union


def clip_mask(tray: TrayConfig, shape: tuple[int, int]) -> np.ndarray:
    """The production clip: tray ROI grown by roi_margin_px, as a boolean frame mask."""
    return metrics.roi_mask(shape, pots.clip_region(tray, shape))


def sam_canopy(rasters: Sequence[np.ndarray], tray: TrayConfig, shape: tuple[int, int]) -> tuple[np.ndarray, list[np.ndarray]]:
    """(union canopy, clipped instance rasters), exactly as segment.write_masks builds them."""
    clip = clip_mask(tray, shape)
    clipped = [r & clip for r in rasters]
    return metrics.union_mask(clipped, shape, None), clipped


def gt_pot_rasters(gt: np.ndarray, tray: TrayConfig) -> list[np.ndarray]:
    """The GT split into one raster per pot (each pixel to the pot pots.pot_of gives it); empty pots omitted."""
    ys, xs = np.nonzero(gt)
    if len(xs) == 0:
        return []
    if tray.pot_grid.cells:  # per-cell overrides make the lookup non-separable
        ids = np.array([pots.pot_of(tray, float(x), float(y)) for x, y in zip(xs, ys)])
        rows, cols = ids[:, 0], ids[:, 1]
    else:
        cols = np.array([pots.pot_of(tray, float(x), tray.roi[1])[1] for x in range(gt.shape[1])])[xs]
        rows = np.array([pots.pot_of(tray, tray.roi[0], float(y))[0] for y in range(gt.shape[0])])[ys]
    out: list[np.ndarray] = []
    for r, c in sorted(set(zip(rows.tolist(), cols.tolist()))):
        sel = (rows == r) & (cols == c)
        m = np.zeros(gt.shape, dtype=bool)
        m[ys[sel], xs[sel]] = True
        out.append(m)
    return out


def compare(gt_full: np.ndarray, sam_rasters: Sequence[np.ndarray], tray: TrayConfig) -> dict:
    """Per-image comparison inside the tray clip region: Dice, IoU, areas, per-pot areas and plant counts."""
    shape = gt_full.shape
    gt = gt_full & clip_mask(tray, shape)
    sam, clipped = sam_canopy(sam_rasters, tray, shape)
    _, gt_pots = pots.assign(tray, gt_pot_rasters(gt, tray))
    _, pix_pots = pots.assign(tray, gt_pot_rasters(sam, tray))
    _, inst_pots = pots.assign(tray, clipped)
    keys = sorted(set(gt_pots) | set(pix_pots) | set(inst_pots))
    return {
        "tray": tray.id,
        "dice": dice(gt, sam),
        "iou": iou(gt, sam),
        "gt_px": metrics.area_px(gt),
        "sam_px": metrics.area_px(sam),
        "tp": int(np.count_nonzero(gt & sam)),
        "gt_plants": len(pots.holding(gt_pots)),
        "sam_plants": len(pots.holding(pix_pots)),
        "inst_plants": len(pots.holding(inst_pots)),
        "pot_pairs": [(gt_pots.get(k, 0), pix_pots.get(k, 0)) for k in keys],
        "inst_pot_pairs": [(gt_pots.get(k, 0), inst_pots.get(k, 0)) for k in keys],
    }


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson r; None with fewer than 2 points or zero variance."""
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _r(v: float | None, nd: int = 4) -> float | None:
    return None if v is None else round(float(v), nd)


def aggregate(rows: Sequence[dict]) -> dict:
    """Aggregates over per-image `compare` rows (n = 0 gives all-None numbers)."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    d, j = np.array([r["dice"] for r in rows]), np.array([r["iou"] for r in rows])
    g, s = np.array([r["gt_px"] for r in rows], float), np.array([r["sam_px"] for r in rows], float)
    tp = sum(r["tp"] for r in rows)
    pooled = 1.0 if g.sum() + s.sum() == 0 else 2.0 * tp / (g.sum() + s.sum())
    r_area = pearson(g, s)
    pairs = [p for r in rows for p in r["pot_pairs"]]
    r_pot = pearson([p[0] for p in pairs], [p[1] for p in pairs])
    ipairs = [p for r in rows for p in r["inst_pot_pairs"]]
    r_ipot = pearson([p[0] for p in ipairs], [p[1] for p in ipairs])
    nz = g > 0
    gp, sp = np.array([r["gt_plants"] for r in rows]), np.array([r["sam_plants"] for r in rows])
    ip = np.array([r["inst_plants"] for r in rows])
    return {
        "n": n,
        "dice_mean": _r(d.mean()), "dice_median": _r(np.median(d)), "dice_min": _r(d.min()),
        "iou_mean": _r(j.mean()), "iou_median": _r(np.median(j)), "iou_min": _r(j.min()),
        "pooled_dice": _r(pooled),
        "area_pearson_r": _r(r_area), "area_r2": _r(None if r_area is None else r_area**2),
        "area_mape": _r(np.mean(np.abs(s[nz] - g[nz]) / g[nz]) if nz.any() else None),
        "pot_n": len(pairs), "pot_pearson_r": _r(r_pot),
        "plant_exact_frac": _r(np.mean(gp == sp)), "plant_mae": _r(np.mean(np.abs(gp - sp))),
        "inst_pot_pearson_r": _r(r_ipot),
        "inst_plant_exact_frac": _r(np.mean(gp == ip)), "inst_plant_mae": _r(np.mean(np.abs(gp - ip))),
    }


def breakdowns(rows: Sequence[dict]) -> dict:
    """overall + by tray (A, B) + by growth stage (small/large: split at the median GT area)."""
    med = float(np.median([r["gt_px"] for r in rows])) if rows else 0.0
    return {
        "overall": aggregate(rows),
        "by_tray": {t: aggregate([r for r in rows if r["tray"] == t]) for t in sorted({r["tray"] for r in rows})},
        "by_stage": {
            "median_gt_px": round(med),
            "small": aggregate([r for r in rows if r["gt_px"] < med]),
            "large": aggregate([r for r in rows if r["gt_px"] >= med]),
        },
    }


def overlay(img_bgr: np.ndarray, gt: np.ndarray, sam: np.ndarray) -> np.ndarray:
    """Frame with TP green, FP red (SAM only), FN blue (GT only), blended over a dimmed frame."""
    out = (img_bgr * 0.5).astype(np.float32)
    for m, bgr in ((gt & sam, (0, 255, 0)), (sam & ~gt, (0, 0, 255)), (gt & ~sam, (255, 0, 0))):
        out[m] = out[m] * 0.3 + np.array(bgr, np.float32) * 0.7
    return out.astype(np.uint8)


def write_overlay(path: Path, jpg: Path, gt: np.ndarray, sam: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay(cv2.imread(str(jpg)), gt, sam))


def evaluate(
    pairs: Sequence[tuple[str, Path, Path]],
    cfg: TraysConfig,
    rasters_for: Callable[[str, Path], list[np.ndarray] | None],
) -> list[dict]:
    """One `compare` row per pair (+ frame, stem). `rasters_for(stem, jpg)` returns the SAM instance rasters, or
    None to skip the frame (uncached under --cached-only, or a failed call)."""
    by_id = {t.id: t for t in cfg.trays}
    rows: list[dict] = []
    for stem, jpg, png in pairs:
        rasters = rasters_for(stem, jpg)
        if rasters is None:
            continue
        rows.append({"frame": frame_id(stem), "stem": stem, **compare(load_gt(png), rasters, by_id[stem[0]])})
    return rows


def compact(row: dict) -> dict:
    keys = ("frame", "tray", "dice", "iou", "gt_px", "sam_px", "gt_plants", "sam_plants", "inst_plants")
    return {k: (round(row[k], 4) if isinstance(row[k], float) else row[k]) for k in keys}


def worst(rows: Sequence[dict], k: int = 3) -> list[dict]:
    return sorted(rows, key=lambda r: r["dice"])[:k]


def build_report(
    rows: Sequence[dict], baseline_rows: Sequence[dict], *, n_gt: int, n_cached: int, n_called: int,
    cost_usd: float, partial: bool, generated_at: str, worst_overlays: dict[str, str] | None = None,
) -> dict:
    """The committed JSON: aggregates for SAM and the HSV baseline, compact per-image rows, provenance."""
    ov = worst_overlays or {}
    return {
        "model": SAM_MODEL, "prompt": PROMPT, "caption": CAPTION,
        "n_images": len(rows), "n_gt": n_gt, "n_cached": n_cached, "n_called": n_called,
        "cost_usd": round(cost_usd, 4), "partial": partial, "generated_at": generated_at,
        "sam": breakdowns(rows),
        "baseline_hsv": {"label": BASELINE_LABEL, **breakdowns(baseline_rows)},
        "worst": [{**compact(r), "overlay": ov.get(r["stem"])} for r in worst(rows)],
        "per_image": [compact(r) for r in rows],
    }
