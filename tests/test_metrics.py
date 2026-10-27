from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from backend.perception import metrics


def test_union_roi_area():
    a = np.zeros((20, 20), bool)
    a[2:6, 2:6] = True
    b = np.zeros((20, 20), bool)
    b[4:8, 4:8] = True
    u = metrics.union_mask([a, b], (20, 20), None)
    assert metrics.area_px(u) == 16 + 16 - 4
    assert metrics.area_cm2(100, 0.01) == pytest.approx(1.0)


def test_roi_clip_exact():
    m = np.ones((10, 10), bool)
    assert metrics.area_px(metrics.union_mask([m], (10, 10), (2, 3, 4, 5))) == 20


def test_exg_and_chlorosis():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:5] = (0, 200, 0)  # pure green (BGR)
    img[5:] = (0, 220, 220)  # yellow, hue 60 deg
    green = np.zeros((10, 10), bool)
    green[:5] = True
    yellow = ~green
    assert metrics.exg(img, green) == pytest.approx(2.0)  # g=1, r=b=0
    assert metrics.exg(img, yellow) == pytest.approx(0.5)  # r=g=.5, b=0
    assert metrics.chlorosis_frac(img, green) == 0.0
    assert metrics.exg(img, np.zeros((10, 10), bool)) is None


def test_chlorosis_band():
    img = np.zeros((4, 4, 3), np.uint8)
    img[:] = (0, 200, 200)  # BGR -> hue 60 deg: outside 20-45
    allm = np.ones((4, 4), bool)
    assert metrics.chlorosis_frac(img, allm) == 0.0
    img[:2] = (0, 165, 230)  # r=230,g=165 -> hue ~ 41 deg, S=1
    assert metrics.chlorosis_frac(img, allm) == pytest.approx(0.5)


def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    for m in (rng.random((7, 9)) > 0.5, np.zeros((3, 3), bool), np.ones((3, 3), bool)):
        assert (metrics.rle_decode(metrics.rle_encode(m)) == m).all()


def test_growth_rate_exponential():
    t0 = datetime(2016, 1, 1, 12, tzinfo=UTC)
    ts = [t0 + timedelta(hours=h) for h in range(0, 30, 3)]
    areas = [1000 * np.exp(0.3 * (t - t0).total_seconds() / 86400) for t in ts]
    at = ts[-1]
    assert metrics.growth_rate_24h(ts, areas, at) == pytest.approx(0.3, rel=1e-6)
    assert metrics.growth_rate_24h(ts[:2], areas[:2], ts[1]) is None
