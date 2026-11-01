"""Frame image and overlay endpoints (M2). The rest of the frames API (shapes in API.md) is M3b."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response

from backend.api import deps
from backend.config import frame_file
from backend.perception.metrics import rle_decode
from backend.timeutil import iso_z

router = APIRouter()


def frame_item(r: sqlite3.Row) -> dict:
    base = f"/trays/{r['tray_id']}/frames/{r['id']}"
    return {
        "id": r["id"],
        "tray_id": r["tray_id"],
        "ts": r["ts"],
        "source": r["source"],
        "calib_flag": r["calib_flag"],
        "cm2_per_px": r["cm2_per_px"],
        "image_url": f"{base}/image",
        "overlay_url": f"{base}/overlay",
    }


@router.get("/trays/{tray_id}/frames")
def list_frames(
    tray_id: str,
    limit: int = Query(50, ge=1, le=500),
    cursor: str | None = None,
    as_of: datetime = Depends(deps.get_as_of),
    conn: sqlite3.Connection = Depends(deps.get_conn),
) -> dict:
    """Newest first, ts <= as_of. `cursor` is the ts of the last item of the previous page (exclusive)."""
    deps.require_tray(tray_id)
    hi = iso_z(as_of)
    q, args = "SELECT * FROM frames WHERE tray_id = ? AND ts <= ?", [tray_id, hi]
    if cursor:
        q += " AND ts < ?"
        args.append(cursor)
    rows = conn.execute(q + " ORDER BY ts DESC LIMIT ?", [*args, limit + 1]).fetchall()
    items = [frame_item(r) for r in rows[:limit]]
    return {"items": items, "next_cursor": items[-1]["ts"] if len(rows) > limit else None}


def _frame(conn: sqlite3.Connection, tray_id: str, fid: str, as_of: datetime) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM frames WHERE id = ? AND tray_id = ?", (fid, tray_id)).fetchone()
    if row is None or row["ts"] > iso_z(as_of) or not frame_file(row["path"]).is_file():
        raise HTTPException(404, "frame not found")
    return row


@router.get("/trays/{tray_id}/frames/{fid}/image")
def frame_image(tray_id: str, fid: str, as_of: datetime = Depends(deps.get_as_of)) -> Response:
    conn = deps.open_db()
    try:
        row = _frame(conn, tray_id, fid, as_of)
    finally:
        conn.close()
    return Response(frame_file(row["path"]).read_bytes(), media_type="image/jpeg")


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
def frame_overlay(tray_id: str, fid: str, as_of: datetime = Depends(deps.get_as_of)) -> Response:
    conn = deps.open_db()
    try:
        row = _frame(conn, tray_id, fid, as_of)
        m = conn.execute("SELECT rle FROM masks WHERE frame_id = ? AND kind = 'canopy'", (fid,)).fetchone()
    finally:
        conn.close()
    if m is None:
        raise HTTPException(404, "frame has no canopy mask yet")
    img = cv2.imread(str(frame_file(row["path"])))
    mask = rle_decode(json.loads(m["rle"]))
    if mask.shape != img.shape[:2]:  # the demo bundle ships downscaled frames and masks of the same (smaller) size
        mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    _ok, png = cv2.imencode(".png", render_overlay(img, mask))
    return Response(png.tobytes(), media_type="image/png")
