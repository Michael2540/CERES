"""Perception fixes after the M6 validation: pot-edge rounding, merged-rosette splitting, the SAM colour trim
and the never-pay-twice guard for derived frames. All offline; the SAM backend is a stub that fails if called."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from test_pipeline import CFG

from backend.db import init_db
from backend.perception import pots, segment, trim
from backend.perception.segment import MissingRaw, estimate_cost_usd, frame_source, segment_frames

A = CFG.by_id("A")


def _rect(x0, y0, x1, y1, shape=(768, 1024)) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


# ------------------------------------------------------------------ pot edges
@pytest.mark.parametrize("roi", [(65, 15, 905, 740), (10, 7, 903, 741), (0, 0, 1001, 763)])
def test_pot_bounds_and_pot_of_agree_on_every_cell_edge(roi):
    t = A.model_copy(update={"roi": roi})
    for (r, c), (x0, y0, x1, y1) in pots.pot_bounds(t).items():
        for x, y in ((x0, y0), (x1 - 1, y1 - 1)):  # first and last pixel of the half-open cell
            assert pots.pot_of(t, x, y) == (r, c)
        if x1 < roi[0] + roi[2]:
            assert pots.pot_of(t, x1, y0) == (r, c + 1)  # the next pixel belongs to the neighbour
        if y1 < roi[1] + roi[3]:
            assert pots.pot_of(t, x0, y1) == (r + 1, c)
    lab = pots.pot_labels(t, (768, 1024))
    ys, xs = np.mgrid[0:768:37, 0:1024:41]
    for y, x in zip(ys.ravel(), xs.ravel()):
        r, c = pots.pot_of(t, float(x), float(y))
        assert lab[y, x] == r * 5 + c


# ------------------------------------------------------------------ merged rosettes
def test_split_by_pot_cuts_a_merged_pair_and_keeps_small_overhangs():
    merged = _rect(200, 60, 300, 140)  # pot00 (x < 246) and pot01 (x >= 246): 46 vs 54 columns
    parts = pots.split_by_pot(A, [merged])
    assert len(parts) == 2 and sum(int(p.sum()) for p in parts) == int(merged.sum())
    ids, area = pots.assign(A, parts)
    assert sorted(ids) == ["A:pot00", "A:pot01"] and all(v >= pots.POT_MIN_PX for v in area.values())
    overhang = _rect(100, 60, 252, 140)  # 6 of 152 columns in pot01: stays whole in pot00
    (whole,) = pots.split_by_pot(A, [overhang])
    assert np.array_equal(whole, overhang) and pots.assign(A, [whole])[0] == ["A:pot00"]
    mid = _rect(100, 60, 260, 140)  # 14 of 160 columns in pot01 (< 25 %): overhang goes to the main cell
    assert len(pots.split_by_pot(A, [mid])) == 1
    assert pots.split_by_pot(A, []) == []


def test_split_folds_below_threshold_cells_into_the_largest_part():
    m = _rect(150, 60, 260, 140) | _rect(150, 190, 330, 250)  # pot00 big, pot01 sliver, pot10/11 pieces
    parts = pots.split_by_pot(A, [m])
    assert sum(int(p.sum()) for p in parts) == int(m.sum())  # nothing lost, nothing double counted
    assert all(int(p.sum()) >= pots.POT_MIN_PX for p in parts)


# ------------------------------------------------------------------ colour trim
def test_trim_drops_non_plant_fringe_and_refills_holes():
    img = np.full((120, 160, 3), 20, np.uint8)
    green = cv2.cvtColor(np.uint8([[[60, 200, 150]]]), cv2.COLOR_HSV2BGR)[0, 0]
    img[30:90, 30:90] = green
    img[58:62, 58:62] = 20  # a specular hole inside the leaf
    inst = _rect(20, 20, 100, 100, (120, 160))  # SAM mask: leaf plus a 10 px fringe
    (out,) = trim.trim_instances(img, [inst])
    assert out[60, 60] and not out[22, 22] and not out[95, 95]  # hole refilled, fringe gone
    assert not (out & ~inst).any()  # a trim never grows a mask


# ------------------------------------------------------------------ never pay twice
class _Explode:
    name, model = "api", "sam-3.1"

    async def acquire(self, *a, **k):
        raise AssertionError("SAM must not be called")

    adecode = acquire


def _derived_frame(conn, tmp_path):
    p = tmp_path / "B_20160106T100000.jpg"
    cv2.imwrite(str(p), np.full((20, 20, 3), 70, np.uint8))
    conn.execute(
        "INSERT INTO frames(id, tray_id, ts, path, cm2_per_px, calib_flag, source, fault_hash) VALUES"
        " ('B:2016-01-06T10:00:00Z','B','2016-01-06T10:00:00Z',?,0.01,'n','fault_transform','h1')",
        (str(p),),
    )
    conn.commit()
    return conn.execute("SELECT id, tray_id, ts, path, cm2_per_px, fault_hash FROM frames").fetchone()


def test_missing_raw_file_never_reaches_sam(tmp_path, monkeypatch):
    conn = init_db(":memory:")
    f = _derived_frame(conn, tmp_path)
    monkeypatch.setattr(segment, "RAW_DIR", tmp_path / "no_raw")
    with pytest.raises(MissingRaw):
        frame_source(f)
    assert estimate_cost_usd(conn, [f], tmp_path / "cache") == 0  # not billable either
    stats = segment_frames(conn, CFG, _Explode(), cache_dir=tmp_path / "cache")  # type: ignore[arg-type]
    assert stats["missing_raw"] == 1 and stats["segmented"] == 0 and stats["errors"] == 0


def test_frame_source_dataset_frame_needs_no_raw_file(tmp_path):
    conn = init_db(":memory:")
    f = _derived_frame(conn, tmp_path)
    conn.execute("UPDATE frames SET fault_hash = NULL")
    f = conn.execute("SELECT id, tray_id, ts, path, cm2_per_px, fault_hash FROM frames").fetchone()
    assert frame_source(f, tmp_path / "none")[1:] == (None, None)  # raw frames are keyed on themselves
