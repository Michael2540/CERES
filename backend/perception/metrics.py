"""Canopy metrics (M2): pure functions on numpy masks and BGR frames.

Masks are boolean arrays shaped (h, w). RLE is a compact JSON-able dict {"size": [h, w], "counts": [...]}:
alternating run lengths over the row-major flattened mask, starting with a run of zeros.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

import cv2
import numpy as np

DEFAULT_CHLOROSIS = {"hue_min": 20.0, "hue_max": 45.0, "sat_min": 0.25, "val_min": 0.2}  # hue in degrees


def rle_encode(mask: np.ndarray) -> dict:
    flat = mask.astype(bool).ravel()
    h, w = mask.shape
    if flat.size == 0:
        return {"size": [h, w], "counts": []}
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    counts = np.diff(bounds).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return {"size": [h, w], "counts": counts}


def rle_decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for n in rle["counts"]:
        if val:
            flat[pos : pos + n] = True
        pos += n
        val = not val
    return flat.reshape(h, w)


def roi_mask(shape: tuple[int, int], roi: tuple[int, int, int, int] | None) -> np.ndarray:
    """Boolean mask of the ROI (x, y, w, h); everything when roi is None."""
    if roi is None:
        return np.ones(shape, dtype=bool)
    x, y, w, h = roi
    m = np.zeros(shape, dtype=bool)
    m[y : y + h, x : x + w] = True
    return m


def union_mask(
    masks: Sequence[np.ndarray], shape: tuple[int, int], roi: tuple[int, int, int, int] | None
) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for m in masks:
        out |= m.astype(bool)
    return out & roi_mask(shape, roi)


def area_px(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def area_cm2(area_px_: float, cm2_per_px: float) -> float:
    return float(area_px_) * cm2_per_px


def exg(img_bgr: np.ndarray, mask: np.ndarray) -> float | None:
    """Mean excess green 2g - r - b over chromatic-normalised rgb inside the mask (None if empty)."""
    px = img_bgr[mask.astype(bool)].astype(np.float64)
    if px.size == 0:
        return None
    s = px.sum(axis=1)
    s[s == 0] = 1.0
    b, g, r = px[:, 0] / s, px[:, 1] / s, px[:, 2] / s
    return float(np.mean(2 * g - r - b))


def chlorosis_frac(img_bgr: np.ndarray, mask: np.ndarray, band: dict | None = None) -> float | None:
    """Fraction of mask pixels in the yellow band (hue degrees, S and V in 0-1). None if mask empty."""
    band = band or DEFAULT_CHLOROSIS
    m = mask.astype(bool)
    n = np.count_nonzero(m)
    if n == 0:
        return None
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(np.float32) * 2.0
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0
    yellow = (
        (hue >= band["hue_min"])
        & (hue <= band["hue_max"])
        & (sat > band["sat_min"])
        & (val > band["val_min"])
    )
    return float(np.count_nonzero(yellow & m) / n)


def growth_rate_24h(ts_list: Sequence[datetime], area_list: Sequence[float], at_ts: datetime) -> float | None:
    """Least-squares slope of ln(area) per day over (at_ts - 24 h, at_ts]; None with < 3 usable points."""
    lo = at_ts - timedelta(hours=24)
    pts = [(t, a) for t, a in zip(ts_list, area_list) if lo < t <= at_ts and a and a > 0]
    if len(pts) < 3:
        return None
    x = np.array([(t - at_ts).total_seconds() / 86400.0 for t, _ in pts])
    y = np.log([a for _, a in pts])
    if np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])
