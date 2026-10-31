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
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import cv2

from backend.config import ROOT, TrayConfig, TraysConfig
from backend.perception.sam_client import CACHE_DIR, cache_path
from backend.sim.faults import FAULTS_VERSION, dim, timewarp_source_ts, wilt_transform

log = logging.getLogger(__name__)
DERIVED_DIR = ROOT / "data" / "derived"
SRC_DIRNAME = "aled_src"  # sibling of the mission folder: extra real frames, time-warp sources only
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


TIMEWARP_VERSION = "lit-1"  # lamp-hours warp; independent of FAULTS_VERSION, so regenerating B never touches C


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


def _scan(folder: Path, trays: dict[str, TrayConfig]) -> dict[str, dict[datetime, Path]]:
    out: dict[str, dict[datetime, Path]] = {}
    if folder.is_dir():
        for p in sorted(folder.glob("*.jpg")):
            m = _NAME.match(p.name)
            if m and m["tray"] in trays:
                out.setdefault(m["tray"], {})[_parse_ts(m["ts"])] = p
    return out


def _timewarp_src(tray: TrayConfig, ts: datetime, pool: dict[datetime, Path]) -> Path:
    """The real frame (mission or extra source) that stands in for `ts` on a low-light tray."""
    fault_start = _iso(tray.fault_start)
    assert fault_start is not None and tray.fault.growth_factor is not None
    return pool[timewarp_source_ts(ts, fault_start, tray.fault.growth_factor, sorted(pool), tray.lamp)]


def _derive(
    tray: TrayConfig, ts: datetime, raw: dict[datetime, Path], out: Path, pool: dict[datetime, Path] | None = None
) -> tuple[str, str | None]:
    """Write the faulted frame for (tray, ts) to `out`; return (source label, time-warp source file name)."""
    fault_start, f, kind = _iso(tray.fault_start), tray.fault, _kind(tray)
    assert fault_start is not None
    src_name = None
    if kind == "fault_transform":
        before = [t for t in raw if t <= fault_start]
        ref = cv2.imread(str(raw[max(before)])) if before else None  # last real frame <= fault_start
        days = (ts - fault_start).total_seconds() / 86400.0
        img = wilt_transform(cv2.imread(str(raw[ts])), days, f, tuple(tray.roi), ref, tray.roi_margin_px)
    else:
        src = _timewarp_src(tray, ts, pool if pool is not None else raw)
        src_name = src.name
        img = cv2.imread(str(src))
        if f.brightness_factor is not None:
            img = dim(img, f.brightness_factor)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return kind, src_name


def _adopt_cache(old_raw: Path, new_raw: Path, fh: str, cache_dir: Path) -> None:
    """Copy cache entries keyed on the mission frame's own raw file (the pre-aled_src key) to the key of the
    real source frame, for a frame whose derived image did not change. Nothing is billed twice."""
    from backend.perception.segment import PROMPT

    for b in ("api", "hsv"):
        old, new = cache_path(old_raw, PROMPT, b, cache_dir, old_raw, fh), cache_path(new_raw, PROMPT, b, cache_dir, new_raw, fh)
        if old.exists() and not new.exists():
            shutil.copyfile(old, new)


def ingest_folder(
    conn: sqlite3.Connection,
    folder: str | Path,
    cfg: TraysConfig,
    derived_dir: Path = DERIVED_DIR,
    src_folder: Path | None = None,
    cache_dir: Path = CACHE_DIR,
) -> int:
    """Insert frames for every <tray>_<ts>.jpg in folder; returns the number of new rows.

    src_folder (default: `aled_src` next to folder) holds extra real frames that are only candidates for a
    time-warp source; they never become mission frames."""
    folder = Path(folder)
    trays = {t.id: t for t in cfg.trays}
    raw = _scan(folder, trays)
    extra = _scan(src_folder if src_folder is not None else folder.parent / SRC_DIRNAME, trays)
    existing = {r["id"]: (r["fault_hash"], r["src_file"]) for r in conn.execute("SELECT id, fault_hash, src_file FROM frames")}
    seen_sha: set[str] = set()
    inserted = regenerated = adopted = 0
    derived_dir.mkdir(parents=True, exist_ok=True)
    for tid, by_ts in raw.items():
        tray = trays[tid]
        fh, start, kind = fault_hash(tray), _iso(tray.fault_start), _kind(tray)
        pool = {**extra.get(tid, {}), **by_ts}  # union; the mission file wins a same-timestamp clash
        for ts, p in sorted(by_ts.items()):
            fid = f"{tid}:{iso_z(ts)}"
            derived = fh is not None and start is not None and ts >= start
            want_path = _timewarp_src(tray, ts, pool) if derived and kind == "fault_timewarp" else None
            want = want_path.name if want_path else None
            if fid in existing:
                cur_fh, cur_src = existing[fid]
                if not derived or (cur_fh == fh and cur_src == want):
                    continue
                if cur_fh == fh and cur_src is None and want is not None:
                    # rows from before aled_src: the old source came from the mission pool alone
                    old = _timewarp_src(tray, ts, by_ts)
                    if old.name == want:  # same derived image: keep its masks, move its cache entry
                        _adopt_cache(p, want_path, fh, cache_dir)
                        conn.execute("UPDATE frames SET src_file = ? WHERE id = ?", (want, fid))
                        existing[fid] = (fh, want)
                        adopted += 1
                        continue
                out = derived_dir / p.name  # stale derived frame: regenerate, drop its masks
                source, src_name = _derive(tray, ts, by_ts, out, pool)
                conn.execute(
                    "UPDATE frames SET path = ?, source = ?, fault_hash = ?, src_file = ? WHERE id = ?",
                    (_rel(out), source, fh, src_name, fid),
                )
                conn.execute("DELETE FROM masks WHERE frame_id = ?", (fid,))
                existing[fid] = (fh, src_name)
                regenerated += 1
                continue
            sha = _sha256(p)
            if sha in seen_sha:
                continue
            seen_sha.add(sha)
            src_name = None
            if derived:
                out = derived_dir / p.name
                (source, src_name), path = _derive(tray, ts, by_ts, out, pool), out
            else:
                source, path = "dataset", p
            conn.execute(
                "INSERT INTO frames(id, tray_id, ts, path, cm2_per_px, calib_flag, source, fault_hash, src_file)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    fid,
                    tid,
                    iso_z(ts),
                    _rel(path),
                    tray.cm2_per_px,
                    tray.calib_flag,
                    source,
                    fh if derived else None,
                    src_name,
                ),
            )
            existing[fid] = (fh if derived else None, src_name)
            inserted += 1
    conn.commit()
    if regenerated or adopted:
        log.info("regenerated %d stale derived frames, kept %d with an unchanged source", regenerated, adopted)
    return inserted
