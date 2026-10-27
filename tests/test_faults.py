import itertools
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.config import ROOT, FaultParams, load_trays
from backend.perception.sam_client import HsvBackend
from backend.sim.faults import dim, plant_mask, scale_at, timewarp_source_ts, wilt_transform


def test_timewarp_picks_closest_real_frame():
    fs = datetime(2016, 1, 3, 8, tzinfo=UTC)
    avail = [fs + timedelta(hours=h) for h in range(-10, 60, 6)]
    ts = fs + timedelta(hours=40)  # target = fs + 16 h
    assert timewarp_source_ts(ts, fs, 0.4, avail) == fs + timedelta(hours=14)  # |14-16| < |20-16|
    assert timewarp_source_ts(fs - timedelta(hours=1), fs, 0.4, avail) == fs - timedelta(hours=1)
    # never later than ts
    assert timewarp_source_ts(fs + timedelta(hours=2), fs, 0.9, avail) <= fs + timedelta(hours=2)


def test_dim():
    assert dim(np.full((2, 2, 3), 100, np.uint8), 0.6)[0, 0, 0] == 60


def _plants(leaf=(30, 170, 60), size=20) -> np.ndarray:
    img = np.full((160, 160, 3), (10, 15, 20), np.uint8)  # dark soil (BGR)
    for cx, cy in ((40, 40), (110, 110)):
        img[cy - size : cy + size, cx - size : cx + size] = leaf
    return img


P = FaultParams(shrink_per_day=0.05, s_min=0.55, desat_per_day=0.08)


def _area(img: np.ndarray) -> int:
    return int(plant_mask(img).sum())


def test_wilt_shrinks_without_holes_and_monotonically():
    img = _plants()
    areas = [_area(wilt_transform(img, d, P)) for d in (0, 1, 2, 4, 8)]
    assert areas[0] == _area(img)
    assert all(a > b for a, b in itertools.pairwise(areas))
    assert areas[-1] == pytest.approx(areas[0] * scale_at(8, P) ** 2, rel=0.1)
    assert (wilt_transform(img, 3, P) == wilt_transform(img, 3, P)).all()  # deterministic
    out = wilt_transform(img, 2, P)  # plant stays one solid blob around its centroid: no holes
    n, _ = cv2.connectedComponents(plant_mask(out).astype(np.uint8))
    assert n == 3  # background + 2 plants


def test_wilt_target_is_fault_start_footprint_even_if_plant_grew():
    ref, grown = _plants(size=15), _plants(size=25)  # raw plant keeps growing after fault_start
    a_ref = _area(ref)
    for d in (1, 2, 4, 8):
        a = _area(wilt_transform(grown, d, P, ref_img=ref))
        assert a <= a_ref * scale_at(d, P) ** 2 * 1.08
    a = [_area(wilt_transform(grown, d, P, ref_img=ref)) for d in (1, 2, 4, 8)]
    assert all(x >= y * 0.99 for x, y in itertools.pairwise(a))


def test_wilt_dulls_colour_subtly_then_obviously():
    img = _plants()
    mean = lambda o: o[plant_mask(img)].reshape(-1, 3).mean(axis=0)
    diffs = [np.abs(mean(wilt_transform(img, d, P)) - mean(img)).sum() for d in (1, 4)]
    assert diffs[0] < diffs[1] and diffs[1] > 15


def _real_b():
    files = sorted((ROOT / "data" / "raw" / "aled").glob("B_*.jpg"))
    files = [f for f in files if not f.stem.endswith(("_mask", "_gt"))]
    if len(files) < 20:
        pytest.skip("ALED tray B frames not present")
    return files


def test_real_b_canopy_area_non_increasing_after_fault_start():
    cfg = load_trays().by_id("B")
    start = datetime.fromisoformat(cfg.fault_start)
    files = _real_b()
    ts = [datetime.strptime(f.stem[2:], "%Y%m%dT%H%M%S").replace(tzinfo=UTC) for f in files]
    ref = cv2.imread(str(files[max(i for i, t in enumerate(ts) if t <= start)]))
    after = [(t, f) for t, f in zip(ts, files) if t >= start][::8]  # every 8th real frame
    hsv = HsvBackend()
    areas = []
    for t, f in after:
        d = (t - start).total_seconds() / 86400
        out = wilt_transform(cv2.imread(str(f)), d, cfg.fault, tuple(cfg.roi), ref)
        tmp = Path(tempfile.mkdtemp()) / "x.png"
        cv2.imwrite(str(tmp), out)
        areas.append((d, sum(m.raster().sum() for m in hsv.segment(tmp, "plant"))))
    a = [x[1] for x in areas]
    assert all(y <= x * 1.01 for x, y in itertools.pairwise(a)), (
        areas
    )  # non-increasing (1% mask noise at the plateau)
    day1 = min(areas, key=lambda x: abs(x[0] - 1))
    assert day1[1] >= 0.85 * a[0], areas
