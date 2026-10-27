"""Simulated faults on real ALED frames (PLAN 0.3, M1b). Pure, deterministic, parameters from trays.yaml.

Bump FAULTS_VERSION whenever the transforms change; it is part of every derived frame's fault_hash.

Tray B, wilting (`wilt_transform`), a drought seen from above:
- Plant pixels are an HSV threshold covering green AND yellow hues (OpenCV hue 15-90, S >= 50, V >= 90).
- Reference = the last real frame at or before fault_start (`ref_img`). The plants there are the size
  every later frame is measured against: a wilting plant stops growing, and the fault runs on real frames
  whose plants keep growing, so using the raw plants would let the canopy regrow and overlap.
- For days d since fault_start the scale is s(d) = max(s_min, 1 - shrink_per_day * d). Each reference plant
  (connected component of the closed plant mask) is shrunk toward its own centroid by exactly s(d) with an
  affine warp (soft alpha edge, no erosion, so no holes), dulled, and pasted onto a soil background.
  Transformed plant area is therefore area_at_fault_start * s(d)^2: monotone non-increasing in d, constant
  once s_min is reached (9 days with the defaults), and never larger than at fault_start.
- Background: the current raw frame everywhere except under its own plants (dilated 5 px); under them the
  soil comes from the reference frame with its plants inpainted away (cv2.inpaint TELEA), so the soil, tray
  and sensor noise stay real and no leaf colour bleeds into the fill.
- Colour: t = min(1, desat_per_day * d). Plant pixels blend toward grey-olive: hue toward 19 (38 degrees,
  inside the chlorosis band), saturation x (1 - 0.35 t), value x (1 - 0.15 t). Subtle on day 1, obvious by
  day 3-4. The dulled plant stays inside the HSV plant mask, so area metrics do not collapse on colour.
- days = 0 returns the frame unchanged.

Tray C, low light (`timewarp_source_ts`): from fault_start the frame at time t is the real frame closest to
fault_start + (t - fault_start) * growth_factor (never later than t); its brightness is then scaled by
brightness_factor (`dim`). Real pixels, simulated growth lag.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import cv2
import numpy as np

from backend.config import FaultParams

FAULTS_VERSION = "wilt-3"
PLANT_LO = (15, 50, 90)  # OpenCV HSV: hue 30-180 degrees, covers yellow-green to green
PLANT_HI = (90, 255, 255)
TARGET_HUE = 19.0  # OpenCV units, 38 degrees
MIN_PLANT_PX = 150
JOIN_PX = 15


def plant_mask(img_bgr: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Boolean plant mask (HSV green-to-yellow threshold), restricted to roi = (x, y, w, h) when given."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(PLANT_LO), np.array(PLANT_HI)) > 0
    if roi is not None:
        x, y, w, h = roi
        keep = np.zeros_like(m)
        keep[y : y + h, x : x + w] = True
        m &= keep
    return m


def scale_at(days: float, params: FaultParams) -> float:
    return max(
        params.s_min if params.s_min is not None else 0.55, 1.0 - (params.shrink_per_day or 0.0) * days
    )


def _dull(img_bgr: np.ndarray, t: float) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] += (TARGET_HUE - hsv[..., 0]) * t
    hsv[..., 1] *= 1.0 - 0.35 * t
    hsv[..., 2] *= 1.0 - 0.15 * t
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def wilt_transform(
    img_bgr: np.ndarray,
    days_since_fault: float,
    params: FaultParams,
    roi: tuple[int, int, int, int] | None = None,
    ref_img: np.ndarray | None = None,
) -> np.ndarray:
    """Shrink and dull the reference plants of a frame by the time since fault_start (tray B, module doc).

    ref_img is the last real frame at or before fault_start (the frame itself when None); roi restricts
    where reference plants are looked for.
    """
    if days_since_fault <= 0:
        return img_bgr.copy()
    ref = ref_img if ref_img is not None else img_bgr
    s = scale_at(days_since_fault, params)
    t = min(1.0, (params.desat_per_day or 0.0) * days_since_fault)
    h, w = ref.shape[:2]

    ref_mask = plant_mask(ref, roi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (JOIN_PX, JOIN_PX))
    joined = cv2.morphologyEx(ref_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    n, labels, _, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)

    cur_hole = cv2.dilate(plant_mask(img_bgr).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ref_hole = cv2.dilate(ref_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) * 255
    soil = cv2.inpaint(ref, ref_hole, 5, cv2.INPAINT_TELEA)
    out = img_bgr.copy()
    out[cur_hole] = soil[cur_hole]  # same-size frames: soil under the current plants, from the reference
    out = out.astype(np.float32)

    dulled = _dull(ref, t)
    for i in range(1, n):
        own = ((labels == i) & ref_mask).astype(np.uint8)
        ys, xs = np.nonzero(own)
        if len(xs) < MIN_PLANT_PX:
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        own = cv2.dilate(own, np.ones((3, 3), np.uint8))  # keep the anti-aliased leaf fringe
        mat = np.array([[s, 0, (1 - s) * cx], [0, s, (1 - s) * cy]], np.float32)
        alpha = cv2.warpAffine(own.astype(np.float32), mat, (w, h), flags=cv2.INTER_LINEAR)[..., None]
        warped = cv2.warpAffine(dulled, mat, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)
        out = out * (1 - alpha) + warped * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def timewarp_source_ts(
    ts: datetime, fault_start: datetime, growth_factor: float, available_ts: Sequence[datetime]
) -> datetime:
    """The real frame time that stands in for `ts` on a low-light tray (see module docstring)."""
    if ts < fault_start:
        return ts
    target = fault_start + (ts - fault_start) * growth_factor
    candidates = [a for a in available_ts if a <= ts]
    if not candidates:
        return ts
    return min(candidates, key=lambda a: (abs((a - target).total_seconds()), a))


def dim(img_bgr: np.ndarray, brightness_factor: float) -> np.ndarray:
    """Scale frame brightness (lamp dimming)."""
    return np.clip(img_bgr.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
