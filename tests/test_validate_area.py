"""Area validation (M6): Dice/IoU, GT loading, per-pot pairing, dry-run / cached-only CLI, the endpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api import evaluation
from backend.app import app
from backend.config import load_trays
from backend.eval import validate_area as va
from backend.perception import pots
from backend.perception.sam_client import ApiBackend, cache_path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_area as cli

TRAY_A = next(t for t in load_trays().trays if t.id == "A")


def box(shape, y0, y1, x0, x1):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_dice_iou_identical_disjoint_half():
    a = box((20, 20), 0, 10, 0, 10)
    assert va.dice(a, a) == 1.0 and va.iou(a, a) == 1.0
    b = box((20, 20), 10, 20, 10, 20)
    assert va.dice(a, b) == 0.0 and va.iou(a, b) == 0.0
    c = box((20, 20), 0, 10, 5, 15)  # half overlap: |A&C|=50, |A|=|C|=100
    assert va.dice(a, c) == pytest.approx(0.5) and va.iou(a, c) == pytest.approx(1 / 3)
    z = np.zeros((5, 5), bool)
    assert va.dice(z, z) == 1.0 and va.iou(z, z) == 1.0


def test_load_gt_non_black_is_plant(tmp_path):
    img = np.zeros((6, 8, 4), np.uint8)
    img[..., 3] = 255
    img[1, 2, :3] = (0, 0, 7)
    img[3, 4, :3] = (200, 10, 0)
    cv2.imwrite(str(tmp_path / "g_gt.png"), img)
    gt = va.load_gt(tmp_path / "g_gt.png")
    assert gt.dtype == bool and gt.shape == (6, 8) and gt.sum() == 2 and gt[1, 2] and gt[3, 4]
    with pytest.raises(FileNotFoundError):
        va.load_gt(tmp_path / "missing.png")


def test_frame_id():
    assert va.frame_id("A_20151214T125406") == "A:2015-12-14T12:54:06Z"


def test_per_pot_pairing_and_counts():
    shape = (768, 1024)
    b = pots.pot_bounds(TRAY_A)
    gt = np.zeros(shape, bool)
    sam = []
    for (r, c), n in {(0, 0): 30, (1, 2): 20}.items():  # two pots hold a plant (>= POT_MIN_PX)
        x0, y0, x1, y1 = b[(r, c)]
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        gt[cy - n : cy + n, cx - n : cx + n] = True
        m = np.zeros(shape, bool)
        m[cy - n : cy + n, cx - n : cx + n + 5] = True  # SAM slightly wider
        sam.append(m)
    x0, y0, x1, y1 = b[(3, 4)]  # a speck in a third pot: below POT_MIN_PX, SAM only
    sp = np.zeros(shape, bool)
    sp[y0 + 50 : y0 + 55, x0 + 50 : x0 + 55] = True
    sam.append(sp)
    row = va.compare(gt, sam, TRAY_A)
    assert row["gt_plants"] == 2 and row["sam_plants"] == 2
    pairs = row["pot_pairs"]
    assert len(pairs) == 3 and sorted(g for g, _ in pairs) == [0, 1600, 3600]
    assert all(s >= g for g, s in pairs)
    agg = va.aggregate([row])
    assert agg["n"] == 1 and agg["plant_exact_frac"] == 1.0 and agg["pot_n"] == 3 and 0 < agg["dice_mean"] < 1


def test_merged_instance_scores_per_pixel_but_not_as_instances():
    """One prediction instance spanning two neighbouring plants (what HSV's closing does to touching rosettes):
    the pixel split credits both pots; the centroid rule puts all of it in one pot."""
    shape = (768, 1024)
    b = pots.pot_bounds(TRAY_A)
    gt = np.zeros(shape, bool)
    for c in (0, 1):
        x0, y0, x1, y1 = b[(0, c)]
        gt[y0 + 40 : y1 - 40, x0 + 40 : x1 - 40] = True
    merged = np.zeros(shape, bool)
    merged[b[(0, 0)][1] + 40 : b[(0, 0)][3] - 40, b[(0, 0)][0] + 40 : b[(0, 1)][2] - 40] = True
    row = va.compare(gt, [merged & gt], TRAY_A)
    assert row["gt_plants"] == 2 and row["sam_plants"] == 2 and row["inst_plants"] == 1
    agg = va.aggregate([row])
    assert agg["plant_exact_frac"] == 1.0 and agg["inst_plant_exact_frac"] == 0.0


def test_aggregate_empty_and_breakdowns():
    assert va.aggregate([]) == {"n": 0}
    rows = [
        {"tray": t, "dice": d, "iou": d / 2, "gt_px": g, "sam_px": g + 1, "tp": g, "gt_plants": 1, "sam_plants": 1,
         "inst_plants": 1, "pot_pairs": [(g, g + 1)], "inst_pot_pairs": [(g, g + 1)], "frame": "x", "stem": "x"}
        for t, d, g in (("A", 0.9, 100), ("A", 0.8, 200), ("B", 0.7, 300), ("B", 0.6, 400))
    ]
    bd = va.breakdowns(rows)
    assert set(bd["by_tray"]) == {"A", "B"} and bd["by_stage"]["small"]["n"] == 2 == bd["by_stage"]["large"]["n"]
    assert bd["overall"]["area_pearson_r"] == pytest.approx(1.0)
    assert [r["dice"] for r in va.worst(rows)] == [0.6, 0.7, 0.8]


def _stage(tmp_path, names=("A_20151214T125406", "A_20151214T162455")):
    for i, n in enumerate(names):
        cv2.imwrite(str(tmp_path / f"{n}.jpg"), np.full((768, 1024, 3), 10 + i, np.uint8))
        cv2.imwrite(str(tmp_path / f"{n}_gt.png"), np.zeros((768, 1024, 4), np.uint8))
    return va.gt_pairs(tmp_path)


def test_dry_run_classifies_cached_vs_uncached_and_never_calls(tmp_path, monkeypatch, capsys):
    pairs = _stage(tmp_path)
    assert len(pairs) == 2
    p = cache_path(pairs[0][1], va.PROMPT, "api")
    monkeypatch.setattr(cli, "va", va)
    existed = p.exists()
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps({"payload": "x"}), encoding="utf-8")
    try:
        assert cli.classify(pairs) == (["A_20151214T125406"], ["A_20151214T162455"])
        boom = pytest.fail
        monkeypatch.setattr(ApiBackend, "acquire", lambda *a, **k: boom("paid call"))
        monkeypatch.setattr(ApiBackend, "_get_client", lambda *a, **k: boom("client built"))
        monkeypatch.setattr(sys, "argv", ["validate_area.py", "--dry-run", "--gt-dir", str(tmp_path)])
        cli.main()
    finally:
        if not existed:
            p.unlink()
    out = capsys.readouterr().out
    assert "2 GT pairs: 1 cached, 1 uncached; estimated cost $0.0025" in out


def test_cached_only_is_partial_and_free(tmp_path, monkeypatch):
    pairs = _stage(tmp_path)
    # the stub payload is replaced by a real-format decode: monkeypatch decode, forbid any call
    async def fake_decode(self, payload, image_path):
        return []

    monkeypatch.setattr(ApiBackend, "adecode", fake_decode)
    monkeypatch.setattr(ApiBackend, "acquire", lambda *a, **k: pytest.fail("paid call"))
    p = cache_path(pairs[0][1], va.PROMPT, "api")
    existed = p.exists()
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps({"payload": "x"}), encoding="utf-8")
    out = tmp_path / "out.json"
    monkeypatch.setattr(va, "OVERLAY_DIR", tmp_path / "ov")
    monkeypatch.setattr(sys, "argv", ["validate_area.py", "--cached-only", "--gt-dir", str(tmp_path), "--out", str(out)])
    try:
        cli.main()
    finally:
        if not existed:
            p.unlink()
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["partial"] is True and rep["n_images"] == 1 and rep["n_called"] == 0 and rep["cost_usd"] == 0
    assert rep["model"] == "sam-3.1" and rep["caption"] == va.CAPTION and rep["baseline_hsv"]["label"]


def test_endpoint_200_and_404(tmp_path, monkeypatch):
    c = TestClient(app)
    monkeypatch.setattr(evaluation, "OUT_PATH", tmp_path / "nope.json")
    assert c.get("/eval/area-validation").status_code == 404
    f = ROOT / "tests" / "fixtures" / "api" / "area_validation.json"
    monkeypatch.setattr(evaluation, "OUT_PATH", f)
    r = c.get("/eval/area-validation", params={"as_of": "2015-12-14T00:00:00Z"})  # not time-gated
    assert r.status_code == 200 and r.json() == json.loads(f.read_text(encoding="utf-8"))
