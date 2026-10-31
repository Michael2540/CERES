"""Virtual camera for CERES: sample ALED frames without the 61.6 GB download.

Pulls only sampled frames via HTTP range requests (``remotezip``).

ALED: Aberystwyth Leaf Evaluation Dataset, Zenodo 168158, CC BY 4.0.
Trays PSI_Tray031..034 map to CERES trays A, B, C, L.

Example: python scripts/fetch_dataset.py --trays A,B --with-masks

--timewarp-sources fetches, into data/raw/aled_src, the real frames that stand in for tray C's low-light
time-warp (backend/sim/faults.py): per mission frame after fault_start, the archive frame nearest to the warp
target (imaging hours, never later than the mission frame). Mission frames (data/raw/aled) are not touched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image as ImageT

ALED_URL = "https://zenodo.org/records/168158/files/images_and_annotations.zip"
TRAY_MAP = {"PSI_Tray031": "A", "PSI_Tray032": "B", "PSI_Tray033": "C", "PSI_Tray034": "L"}
_MEMBER_RE = re.compile(
    r"(PSI_Tray\d+)/tv/(?:(masks|gt)/)?(PSI_Tray\d+)_"
    r"(\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})_top(_mask|_gt)?\.png$"
)


def parse_member(name: str) -> tuple[str, str, datetime] | None:
    """Parse an archive member into (CERES tray, kind, UTC datetime)."""
    if not name or name.endswith("/"):
        return None
    m = _MEMBER_RE.search(name)
    if not m:
        return None
    tray_dir, subdir, _, stamp, suffix = m.groups()
    tray = TRAY_MAP.get(tray_dir)
    if tray is None:
        return None
    kind = ("mask" if suffix == "_mask" or subdir == "masks"
            else "gt" if suffix == "_gt" or subdir == "gt" else "rgb")
    dt = datetime.strptime(stamp, "%Y-%m-%d--%H-%M-%S").replace(tzinfo=UTC)
    return tray, kind, dt


def select_frames(members: list[str], interval_hours: float,
                  per_tray_limit: int | None = None) -> dict[str, list[str]]:
    """Per tray, keep the frame closest to the start of each interval slot."""
    by_tray: dict[str, list[tuple[datetime, str]]] = {}
    for name in members:
        parsed = parse_member(name)
        if parsed is None or parsed[1] != "rgb":
            continue
        by_tray.setdefault(parsed[0], []).append((parsed[2], name))
    step = timedelta(hours=interval_hours)
    out: dict[str, list[str]] = {}
    for tray in sorted(by_tray):
        frames = sorted(by_tray[tray])
        t0 = frames[0][0]
        slots: dict[int, tuple[datetime, str]] = {}
        for dt, name in frames:
            idx = int((dt - t0) // step)
            start = t0 + idx * step
            if idx not in slots or abs(dt - start) < abs(slots[idx][0] - start):
                slots[idx] = (dt, name)
        chosen = [slots[i][1] for i in sorted(slots)]
        out[tray] = chosen[:per_tray_limit] if per_tray_limit is not None else chosen
    return out


def out_name(tray: str, dt: datetime, suffix: str = "") -> str:
    """Return ``<tray>_<YYYYmmddTHHMMSS>[suffix].jpg|.png``."""
    stem = f"{tray}_{dt.strftime('%Y%m%dT%H%M%S')}{suffix}"
    return stem + ".png" if "mask" in suffix or "gt" in suffix else stem + ".jpg"


def _scaled(img: ImageT, width: int, resample: int) -> ImageT:
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), resample)
    return img


_local = threading.local()


def _fetch_bytes(url: str, member: str, retries: int = 3) -> bytes:
    """Read one member, reusing a per-thread RemoteZip so the central directory is fetched once."""
    from remotezip import RemoteZip

    last: Exception | None = None
    for attempt in range(retries):
        try:
            if getattr(_local, "zip", None) is None:
                _local.zip = RemoteZip(url)
            with _local.zip.open(member) as f:
                return f.read()
        except Exception as e:  # noqa: BLE001 - retry any transient error
            last = e
            _local.zip = None
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to fetch {member}: {last}")


def _process(url: str, member: str, dest: Path, width: int, kind: str) -> tuple[int, int]:
    from PIL import Image

    data = _fetch_bytes(url, member)
    img = Image.open(io.BytesIO(data))
    img = _scaled(img.convert("RGB" if kind == "rgb" else img.mode), width,
                  Image.NEAREST if kind in ("mask", "gt") else Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if kind == "rgb":
        img.convert("RGB").save(dest, "JPEG", quality=90)
    else:
        img.save(dest)
    return img.width, img.height


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def list_members(url: str, out: Path) -> list[str]:
    cache = out / ".index.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from remotezip import RemoteZip

    with RemoteZip(url) as z:
        names = z.namelist()
    out.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(names))
    return names


IMAGING_HOURS = range(9, 21)  # UTC hours with a frame every ~13 min; the archive is sparse outside them


def parse_mission_ts(filename: str, tray: str) -> datetime:
    """<tray>_<YYYYmmddTHHMMSS>.jpg -> UTC datetime."""
    return datetime.strptime(filename[len(tray) + 1 : -4], "%Y%m%dT%H%M%S").replace(tzinfo=UTC)


def timewarp_source_jobs(
    names: list[str], tray: str, mission_ts: list[datetime], fault_start: datetime, growth_factor: float, lamp=None
) -> list[tuple[str, datetime]]:
    """(archive member, ts) of every time-warp source that is not already a mission frame, one per distinct
    source. Candidates are the tray's archive frames in imaging hours plus the mission frames themselves
    (all of which the pipeline can pick from), chosen exactly as faults.timewarp_source_ts chooses."""
    from backend.sim.faults import timewarp_source_ts

    arch: dict[datetime, str] = {}
    for n in names:
        parsed = parse_member(n)
        if parsed and parsed[0] == tray and parsed[1] == "rgb" and parsed[2].hour in IMAGING_HOURS:
            arch[parsed[2]] = n
    pool = sorted(set(arch) | set(mission_ts))
    jobs: dict[datetime, str] = {}
    for ts in mission_ts:
        if ts < fault_start:
            continue
        src = timewarp_source_ts(ts, fault_start, growth_factor, pool, lamp)
        if src not in mission_ts and src in arch:
            jobs[src] = arch[src]
    return [(m, t) for t, m in sorted(jobs.items())]


def write_manifest(out: Path, rows: list[dict[str, str | int]]) -> None:
    mp = out / "manifest.csv"
    old: dict[tuple[str, str], dict] = {}
    if mp.exists():
        with open(mp, newline="") as f:
            for r in csv.DictReader(f):
                old[(r["tray"], r["file"])] = r
    for r in rows:
        old[(r["tray"], r["file"])] = r
    with open(mp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tray", "aled_member", "ts_utc", "file",
                                          "sha256", "width", "height"])
        w.writeheader()
        for key in sorted(old):
            w.writerow(old[key])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sample ALED frames via range requests.")
    p.add_argument("--out", default="data/raw/aled")
    p.add_argument("--trays", default="A,B,C,L")
    p.add_argument("--interval-hours", type=float, default=2.0)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--with-masks", action="store_true")
    p.add_argument("--gt", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--timewarp-sources", action="store_true",
                   help="fetch tray C's time-warp source frames into data/raw/aled_src")
    p.add_argument("--mission-dir", default="data/raw/aled", help="mission frames (--timewarp-sources)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args(argv)
    out = Path(a.out)
    if a.out == "data/raw/aled":
        if a.demo:
            out = Path("data/demo/frames")
        elif a.gt:
            out = Path("data/raw/aled_gt")
    width = 512 if a.demo else a.width
    want = {t.strip() for t in a.trays.split(",") if t.strip()}
    if a.timewarp_sources and a.out == "data/raw/aled":
        out = Path("data/raw/aled_src")
    names = list_members(ALED_URL, Path(a.mission_dir) if a.timewarp_sources else out)
    gt_members = [n for n in names if (parse_member(n) or (None, None))[1] == "gt"]
    jobs: list[tuple[str, str, Path, str]] = []  # member, tray, dest, kind
    if a.timewarp_sources:
        from backend.config import load_trays

        for tr in load_trays().trays:
            if tr.id not in want or tr.fault_start is None or tr.fault.growth_factor is None:
                continue
            mission = sorted(
                parse_mission_ts(f.name, tr.id) for f in Path(a.mission_dir).glob(f"{tr.id}_*.jpg")
            )
            start = datetime.fromisoformat(tr.fault_start)
            for member, dt in timewarp_source_jobs(names, tr.id, mission, start, tr.fault.growth_factor, tr.lamp):
                dest = out / out_name(tr.id, dt)
                if not dest.exists():
                    jobs.append((member, tr.id, dest, "rgb"))
    elif a.gt:
        for n in sorted(gt_members):
            parsed = parse_member(n)
            if parsed is None or parsed[0] not in want:
                continue
            tray, _, dt = parsed
            rgb = n.replace("/gt/", "/").replace("_top_gt.png", "_top.png")
            for member, suffix, kind in ((rgb, "", "rgb"), (n, "_gt", "gt")):
                if member not in names and member != n:
                    continue
                dest = out / out_name(tray, dt, suffix)
                if dest.exists():
                    continue
                jobs.append((member, tray, dest, kind))
    else:
        sel = select_frames(names, a.interval_hours, a.limit)
        if a.demo:  # 1 frame per day per tray at 512 px
            sel = {t: [min(g, key=lambda n: parse_member(n)[2]) for _, g in
                       itertools.groupby(sorted(fs, key=lambda n: parse_member(n)[2]),
                                         key=lambda n: parse_member(n)[2].date())]
                   for t, fs in sel.items() if t in want}
        for tray, frames in sorted(sel.items()):
            if tray not in want:
                continue
            for n in frames:
                dt = parse_member(n)[2]
                dest = out / out_name(tray, dt)
                if not dest.exists():
                    jobs.append((n, tray, dest, "rgb"))
                if a.with_masks:
                    mn = n.replace("/tv/", "/tv/masks/").replace("_top.png", "_top_mask.png")
                    md = out / out_name(tray, dt, "_mask")
                    if mn in names and not md.exists():
                        jobs.append((mn, tray, md, "mask"))

    if a.dry_run:
        for member, tray, dest, _ in jobs:
            print(f"{tray} {member} -> {dest}")
        print(f"{len(jobs)} files selected")
        return 0
    rows: list[dict[str, str | int]] = []
    done = skipped = nbytes = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_process, ALED_URL, m, d, width, k): (m, t, d) for m, t, d, k in jobs}
        for fut, (m, t, d) in futs.items():
            try:
                w, h = fut.result()
                dt = parse_member(m)[2]
                rows.append({"tray": t, "aled_member": m, "ts_utc": dt.isoformat(),
                             "file": d.name, "sha256": _sha256(d), "width": w, "height": h})
                done += 1
                nbytes += d.stat().st_size
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f"FAILED {m}: {e}")
                skipped += 1
            if (done + skipped) % 25 == 0:
                print(f"progress: {done + skipped}/{len(jobs)}")
    if rows:
        write_manifest(out, rows)
    print(f"done: {done} downloaded, {skipped} failed, {nbytes} bytes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
