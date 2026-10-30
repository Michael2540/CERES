"""ingest_folder: import <tray>_<YYYYmmddTHHMMSS>.jpg frames into the frames table (M1 / M1b).

Frames of a tray with an image fault, at or after fault_start, are produced by backend/sim/faults.py and
written to data/derived/; the frames row points there and carries source fault_transform (tray B) or
fault_timewarp (tray C) plus a fault_hash of (fault params, fault_start, FAULTS_VERSION). All other frames
point at the raw file with source dataset. Idempotent on frame id and file sha256; when a derived frame's
stored hash no longer matches the current parameters, the file is regenerated, the row updated and the
frame's masks rows deleted so it is segmented again.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import cv2

from backend.config import ROOT, TrayConfig, TraysConfig
from backend.sim.faults import FAULTS_VERSION, dim, timewarp_source_ts, wilt_transform

log = logging.getLogger(__name__)
DERIVED_DIR = ROOT / "data" / "derived"
_NAME = re.compile(r"^(?P<tray>[A-Za-z0-9]+)_(?P<ts>\d{8}T\d{6})\.jpg$")


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)


def iso_z(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _kind(tray: TrayConfig) -> str | None:
    f = tray.fault
    if tray.fault_start is None:
        return None
    if f.shrink_per_day is not None:
        return "fault_transform"
    if f.growth_factor is not None:
        return "fault_timewarp"
    return None


TIMEWARP_VERSION = "wilt-3"  # frozen: tray C's hash must not change when the wilt transform (FAULTS_VERSION) does


def fault_hash(tray: TrayConfig) -> str | None:
    """Hash of everything that determines a derived frame (None when the tray has no image fault). The wilt
    transform (tray B) is versioned by FAULTS_VERSION and the ROI margin; the time-warp (tray C) only changes
    with its own parameters, so regenerating B never invalidates C's masks or SAM cache entries."""
    kind = _kind(tray)
    if kind is None:
        return None
    body = {"start": tray.fault_start, "p": tray.fault.model_dump(), "roi": tray.roi}
    if kind == "fault_transform":
        body |= {"v": FAULTS_VERSION, "margin": tray.roi_margin_px}
    else:
        body |= {"v": TIMEWARP_VERSION}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def _derive(tray: TrayConfig, ts: datetime, raw: dict[datetime, Path], out: Path) -> str:
    """Write the faulted frame for (tray, ts) to `out`; return its source label."""
    fault_start, f, kind = _iso(tray.fault_start), tray.fault, _kind(tray)
    assert fault_start is not None
    if kind == "fault_transform":
        before = [t for t in raw if t <= fault_start]
        ref = cv2.imread(str(raw[max(before)])) if before else None  # last real frame <= fault_start
        days = (ts - fault_start).total_seconds() / 86400.0
        img = wilt_transform(cv2.imread(str(raw[ts])), days, f, tuple(tray.roi), ref, tray.roi_margin_px)
    else:
        src = timewarp_source_ts(ts, fault_start, f.growth_factor, sorted(raw))
        img = cv2.imread(str(raw[src]))
        if f.brightness_factor is not None:
            img = dim(img, f.brightness_factor)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return kind


def ingest_folder(
    conn: sqlite3.Connection, folder: str | Path, cfg: TraysConfig, derived_dir: Path = DERIVED_DIR
) -> int:
    """Insert frames for every <tray>_<ts>.jpg in folder; returns the number of new rows."""
    trays = {t.id: t for t in cfg.trays}
    raw: dict[str, dict[datetime, Path]] = {}
    for p in sorted(Path(folder).glob("*.jpg")):
        m = _NAME.match(p.name)
        if m and m["tray"] in trays:
            raw.setdefault(m["tray"], {})[_parse_ts(m["ts"])] = p
    existing = {r["id"]: r["fault_hash"] for r in conn.execute("SELECT id, fault_hash FROM frames")}
    seen_sha: set[str] = set()
    inserted = regenerated = 0
    derived_dir.mkdir(parents=True, exist_ok=True)
    for tid, by_ts in raw.items():
        tray = trays[tid]
        fh, start = fault_hash(tray), _iso(tray.fault_start)
        for ts, p in sorted(by_ts.items()):
            fid = f"{tid}:{iso_z(ts)}"
            derived = fh is not None and start is not None and ts >= start
            if fid in existing:
                if not derived or existing[fid] == fh:
                    continue
                out = derived_dir / p.name  # stale derived frame: regenerate, drop its masks
                source = _derive(tray, ts, by_ts, out)
                conn.execute(
                    "UPDATE frames SET path = ?, source = ?, fault_hash = ? WHERE id = ?",
                    (_rel(out), source, fh, fid),
                )
                conn.execute("DELETE FROM masks WHERE frame_id = ?", (fid,))
                existing[fid] = fh
                regenerated += 1
                continue
            sha = _sha256(p)
            if sha in seen_sha:
                continue
            seen_sha.add(sha)
            if derived:
                out = derived_dir / p.name
                source, path = _derive(tray, ts, by_ts, out), out
            else:
                source, path = "dataset", p
            conn.execute(
                "INSERT INTO frames(id, tray_id, ts, path, cm2_per_px, calib_flag, source, fault_hash)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (
                    fid,
                    tid,
                    iso_z(ts),
                    _rel(path),
                    tray.cm2_per_px,
                    tray.calib_flag,
                    source,
                    fh if derived else None,
                ),
            )
            existing[fid] = fh if derived else None
            inserted += 1
    conn.commit()
    if regenerated:
        log.info("regenerated %d stale derived frames", regenerated)
    return inserted
