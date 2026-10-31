"""Colour-gated trim of SAM instances (M6 follow-up).

The M6 validation showed SAM masks carry a fringe around leaves (area MAPE 29 % vs 16 % for a plain colour
threshold, up to 2x the annotated area on seedlings). Each instance is cut back to the pixels that pass the
plant-colour test (`sim.faults.plant_mask`, the same test as the HSV backend, with the floor `TRIM_LO`), then
a small morphological close, limited to the instance's own pixels, refills veins and specular holes.

A trim only ever removes pixels of an instance, never adds. The value floor is relative to the frame's own
canopy brightness, so a faulted tray (tray B yellower and duller, tray C dimmed x0.6) is not trimmed harder
than the healthy tray A: `scripts/trim_bias.py` checks that on the mission and the numbers are in PROGRESS.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from backend.sim.faults import PLANT_LO, plant_mask

TRIM_S_MIN = 40  # HSV saturation floor of "plant" (the hue band and PLANT_LO's hue floor are unchanged)
TRIM_V_ABS = 35  # absolute value floor: only pitch-dark pixels fail it
TRIM_V_REL = 0.7  # value floor relative to the frame: this fraction of the 90th-percentile V of the SAM canopy
TRIM_CLOSE_PX = 5  # elliptical closing kernel (0 disables), refills holes inside an instance


def value_floor(img_bgr: np.ndarray, rasters: Sequence[np.ndarray], v_abs: int, v_rel: float) -> int:
    """Value floor of the plant test: max(v_abs, v_rel * p90 of V over the SAM canopy). Tying it to the frame's
    own canopy brightness makes the test invariant to uniform dimming (tray C x0.6) and to the duller wilted
    leaves of tray B, where an absolute floor would read the dimmed frame as having no plant."""
    v = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[..., 2]
    on = np.zeros(v.shape, bool)
    for r in rasters:
        on |= r
    ref = float(np.percentile(v[on], 90)) if on.any() else 255.0
    return int(max(v_abs, v_rel * ref))


def trim_instances(
    img_bgr: np.ndarray,
    rasters: Sequence[np.ndarray],
    s_min: int = TRIM_S_MIN,
    v_abs: int = TRIM_V_ABS,
    v_rel: float = TRIM_V_REL,
    close_px: int = TRIM_CLOSE_PX,
) -> list[np.ndarray]:
    """Each instance raster cut to its plant-coloured pixels, then closed within the original instance."""
    ok = plant_mask(img_bgr, lo=(PLANT_LO[0], s_min, value_floor(img_bgr, rasters, v_abs, v_rel)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)) if close_px > 1 else None
    out: list[np.ndarray] = []
    for r in rasters:
        t = r & ok
        if k is not None and t.any():
            t = (cv2.morphologyEx(t.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0) & r
        out.append(t)
    return out
