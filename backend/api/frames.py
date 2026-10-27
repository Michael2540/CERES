"""Frame image and overlay endpoints (M2). The rest of the frames API (shapes in API.md) is M3b."""

from __future__ import annotations

import json
import sqlite3

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Response

from backend.config import ROOT, load_settings
from backend.db import connect
from backend.perception.metrics import rle_decode

router = APIRouter()


def open_db() -> sqlite3.Connection:
    return connect(ROOT / load_settings().db_path)


def _frame(conn: sqlite3.Connection, tray_id: str, fid: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM frames WHERE id = ? AND tray_id = ?", (fid, tray_id)).fetchone()
    if row is None or not (ROOT / row["path"]).is_file():
        raise HTTPException(404, "frame not found")
    return row


@router.get("/trays/{tray_id}/frames/{fid}/image")
def frame_image(tray_id: str, fid: str) -> Response:
    conn = open_db()
    try:
        row = _frame(conn, tray_id, fid)
    finally:
        conn.close()
    return Response((ROOT / row["path"]).read_bytes(), media_type="image/jpeg")


def render_overlay(
    img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 160, 0)
) -> np.ndarray:
    """Tint the mask and outline it (BGR)."""
    out = img.copy()
    out[mask] = (0.6 * out[mask] + 0.4 * np.array(color)).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


@router.get("/trays/{tray_id}/frames/{fid}/overlay")
def frame_overlay(tray_id: str, fid: str) -> Response:
    conn = open_db()
    try:
        row = _frame(conn, tray_id, fid)
        m = conn.execute("SELECT rle FROM masks WHERE frame_id = ? AND kind = 'canopy'", (fid,)).fetchone()
    finally:
        conn.close()
    if m is None:
        raise HTTPException(404, "frame has no canopy mask yet")
    img = cv2.imread(str(ROOT / row["path"]))
    _ok, png = cv2.imencode(".png", render_overlay(img, rle_decode(json.loads(m["rle"]))))
    return Response(png.tobytes(), media_type="image/png")
