"""Tests for scripts/fetch_dataset.py. No network."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fetch_dataset import out_name, parse_member, select_frames

RGB = "images_and_annotations/PSI_Tray031/tv/PSI_Tray031_2015-12-14--12-54-06_top.png"
MASK = (
    "images_and_annotations/PSI_Tray031/tv/masks/"
    "PSI_Tray031_2015-12-14--12-54-06_top_mask.png"
)
GT = "images_and_annotations/PSI_Tray031/tv/gt/PSI_Tray031_2015-12-14--12-54-06_top_gt.png"


def test_parse_rgb():
    tray, kind, dt = parse_member(RGB)
    assert (tray, kind) == ("A", "rgb")
    assert dt == datetime(2015, 12, 14, 12, 54, 6, tzinfo=UTC)


def test_parse_mask_and_gt():
    assert parse_member(MASK)[:2] == ("A", "mask")
    assert parse_member(GT)[:2] == ("A", "gt")
    assert parse_member(GT)[2].tzinfo == UTC


def test_parse_directory_entry():
    assert parse_member("images_and_annotations/PSI_Tray031/tv/") is None
    assert parse_member("random/file.txt") is None


def test_tray_mapping():
    for code, tray in (("031", "A"), ("032", "B"), ("033", "C"), ("034", "L")):
        name = (
            f"images_and_annotations/PSI_Tray{code}/tv/"
            f"PSI_Tray{code}_2015-12-14--12-00-00_top.png"
        )
        assert parse_member(name)[0] == tray


def _rgb(tray: str, day: int, hour: int, minute: int = 0) -> str:
    return (
        f"images_and_annotations/PSI_Tray{tray}/tv/PSI_Tray{tray}_"
        f"2015-12-{day:02d}--{hour:02d}-{minute:02d}-00_top.png"
    )


def test_select_closest_per_slot_and_skip_empty():
    members = [_rgb("031", 14, 0, 5), _rgb("031", 14, 1, 50), _rgb("031", 14, 5, 10)]
    sel = select_frames(members, 2.0)
    # slots [0:05,2:05) -> 00:05 ; [2:05,4:05) empty ; [4:05,6:05) -> 05:10
    assert sel["A"] == [members[0], members[2]]


def test_select_respects_limit():
    members = [_rgb("032", 14, h) for h in (0, 3, 6, 9)]
    sel = select_frames(members, 1.0, per_tray_limit=2)
    assert len(sel["B"]) == 2


def test_out_name_format():
    dt = datetime(2015, 12, 14, 12, 54, 6, tzinfo=UTC)
    assert out_name("A", dt) == "A_20151214T125406.jpg"
    assert out_name("L", dt, "_mask") == "L_20151214T125406_mask.png"
    assert out_name("B", dt, "_gt") == "B_20151214T125406_gt.png"
