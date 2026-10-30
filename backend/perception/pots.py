"""Per-pot tracking (ALED destructive sampling): pots are emptied in pairs every 5-6 days, so tray-level
canopy area drops without any plant being sick. Each SAM plant instance is assigned to the pot that holds
its centroid, and growth / canopy-drop comparisons use only pots that hold a plant at BOTH ends of the window.

Pot grid: the tray ROI split evenly into cols x rows cells (config `pot_grid`), with optional per-cell bounds
`cells: {"r,c": [x, y, w, h]}` where the even split is off. Rows and columns are 0-based; a pot id is
`<tray>:pot<r><c>` (e.g. `A:pot03`: first row, fourth column). A centroid outside every cell (up to the ROI
margin) is clamped to the nearest cell of the even grid.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np

from backend.config import TrayConfig

POT_MIN_PX = 100  # a pot "holds a plant" when its canopy area is at least this many px


def pot_id(tray_id: str, r: int, c: int) -> str:
    return f"{tray_id}:pot{r}{c}"


def pot_bounds(tray: TrayConfig) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) half-open bounds of every pot: per-cell override, else the even split of the ROI."""
    x, y, w, h = tray.roi
    g = tray.pot_grid
    out: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for r in range(g.rows):
        for c in range(g.cols):
            if (o := g.cells.get(f"{r},{c}")) is not None:
                out[(r, c)] = (o[0], o[1], o[0] + o[2], o[1] + o[3])
            else:
                out[(r, c)] = (
                    x + round(c * w / g.cols), y + round(r * h / g.rows),
                    x + round((c + 1) * w / g.cols), y + round((r + 1) * h / g.rows),
                )
    return out


def pot_of(tray: TrayConfig, cx: float, cy: float) -> tuple[int, int]:
    """The (row, col) of the pot containing (cx, cy); override cells first, then the clamped even grid."""
    g = tray.pot_grid
    for key, o in g.cells.items():
        if o[0] <= cx < o[0] + o[2] and o[1] <= cy < o[1] + o[3]:
            r, c = key.split(",")
            return int(r), int(c)
    x, y, w, h = tray.roi
    c = min(max(int((cx - x) * g.cols // w) if cx >= x else -1, 0), g.cols - 1)
    r = min(max(int((cy - y) * g.rows // h) if cy >= y else -1, 0), g.rows - 1)
    return r, c


def clip_region(tray: TrayConfig, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """ROI grown by roi_margin_px on every side, clamped to the frame, as (x, y, w, h)."""
    x, y, w, h = tray.roi
    m = tray.roi_margin_px
    x0, y0 = max(x - m, 0), max(y - m, 0)
    x1, y1 = min(x + w + m, shape[1]), min(y + h + m, shape[0])
    return x0, y0, x1 - x0, y1 - y0


def assign(
    tray: TrayConfig, rasters: list[np.ndarray]
) -> tuple[list[str | None], dict[str, int]]:
    """Pot id of each (already clipped) instance raster (None when empty), and pot -> union area_px."""
    ids: list[str | None] = []
    union: dict[str, np.ndarray] = {}
    for m in rasters:
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            ids.append(None)
            continue
        r, c = pot_of(tray, float(xs.mean()), float(ys.mean()))
        pid = pot_id(tray.id, r, c)
        ids.append(pid)
        union[pid] = union[pid] | m if pid in union else m.astype(bool)
    return ids, {pid: int(np.count_nonzero(u)) for pid, u in sorted(union.items())}


def load_pots(pots_json: str | None) -> dict[str, int] | None:
    """The `pots_json` column of a canopy row as a mapping (None for rows without per-pot data)."""
    return json.loads(pots_json) if pots_json else None


def holding(pots: Mapping[str, float] | None) -> set[str]:
    return {k for k, v in (pots or {}).items() if v >= POT_MIN_PX}


def plant_count(pots: Mapping[str, float] | None) -> int | None:
    """Number of pots holding a plant (None when the frame has no per-pot data)."""
    return None if pots is None else len(holding(pots))


def tracked_pair(a: Mapping | None, b: Mapping | None) -> tuple[float, float, int] | None:
    """(area of a, area of b, n pots) over the pots holding a plant in BOTH canopy rows, from their `pots`
    mappings. None when either row has no per-pot data (callers fall back to the total area_px)."""
    pa, pb = a.get("pots") if a else None, b.get("pots") if b else None
    if pa is None or pb is None:
        return None
    both = holding(pa) & holding(pb)
    return sum(pa[k] for k in both), sum(pb[k] for k in both), len(both)
