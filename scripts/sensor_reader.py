"""Reads the sensor-hub CSV protocol ("millis,tray_or_air,metric,value") into sensor_readings.

Simulated hub: it has no host clock, so each reading is stamped start + millis (ingest_lines).
Real hub: `--serial PORT` stamps each line with the host UTC time on receipt (needs pyserial, which is
optional and not required by the simulated path).

Beside the raw readings it derives, per tray and per sample time (source="derived"): soil_vwc from
soil_raw (2-point calibration from config/trays.yaml), lamp_on (0/1) and ppfd_est from the lamp schedule.
Rows are inserted with INSERT OR IGNORE, so re-ingesting the same data is a no-op.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import TraysConfig, load_trays
from backend.db import init_db
from backend.science.setpoints import default_setpoints
from backend.timeutil import iso_z, lamp_is_on, parse_ts

VWC_SATURATION = 0.45  # soil_vwc (m3/m3) at wet_raw; dry_raw maps to 0

Sample = tuple[datetime, str, str, float]
_INSERT = "INSERT OR IGNORE INTO sensor_readings(id, ts, tray_id, metric, value, source) VALUES (?,?,?,?,?,?)"


def parse_line(line: str) -> tuple[int, str, str, float] | None:
    """Parse "millis,tray,metric,value"; return None for blank, comment or malformed lines."""
    parts = line.strip().split(",")
    if len(parts) != 4 or parts[0].startswith("#"):
        return None
    try:
        return int(parts[0]), parts[1], parts[2], float(parts[3])
    except ValueError:
        return None


def soil_vwc(raw: float, dry_raw: float, wet_raw: float) -> float:
    """Volumetric water content from the raw probe value by 2-point linear calibration."""
    return round(VWC_SATURATION * (dry_raw - raw) / (dry_raw - wet_raw), 4)  # not clamped


def _row(ts: datetime, tray: str, metric: str, value: float, source: str) -> tuple:
    t = iso_z(ts)
    return f"{tray}:{metric}:{t}", t, tray, metric, value, source


def insert_samples(conn: sqlite3.Connection, samples: Iterable[Sample], cfg: TraysConfig, source: str) -> int:
    """Insert timestamped samples plus the derived rows; returns the number of new rows."""
    rows: list[tuple] = []
    stamps: set[datetime] = set()
    for ts, tray, metric, value in samples:
        rows.append(_row(ts, tray, metric, value, source))
        stamps.add(ts)
        lim = default_setpoints().limit("soil_raw")
        if metric == "soil_raw" and lim and lim[0] <= value <= lim[1]:  # pulled/shorted probe: no derived row
            try:
                cal = cfg.by_id(tray).soil
            except StopIteration:
                continue
            rows.append(_row(ts, tray, "soil_vwc", soil_vwc(value, cal.dry_raw, cal.wet_raw), "derived"))
    ref_ppfd = {t.id: t.ppfd_est for t in cfg.trays}
    for ts in sorted(stamps):
        for tray in cfg.trays:
            on = lamp_is_on(tray.lamp, ts)
            ppfd = tray.ppfd_est
            if tray.treatment == "low_light" and tray.fault_start and ts < parse_ts(tray.fault_start):
                ppfd = ref_ppfd.get(tray.reference_tray_id or "", ppfd)  # dimmed only after fault_start
            rows.append(_row(ts, tray.id, "lamp_on", 1.0 if on else 0.0, "derived"))
            rows.append(_row(ts, tray.id, "ppfd_est", ppfd if on else 0.0, "derived"))
    before = conn.total_changes
    conn.executemany(_INSERT, rows)
    conn.commit()
    return conn.total_changes - before


SENSOR_METRICS = ("temp_c", "rh_pct", "co2_ppm", "soil_raw", "soil_vwc", "lamp_on", "ppfd_est")


def clear_simulated(conn: sqlite3.Connection, start: str | datetime, end: str | datetime) -> int:
    """Delete simulated/derived sensor rows with start <= ts <= end (before re-simulating)."""
    marks = ",".join("?" * len(SENSOR_METRICS))
    cur = conn.execute(
        f"DELETE FROM sensor_readings WHERE source IN ('simulated','derived') AND ts >= ? AND ts <= ? "
        f"AND metric IN ({marks})",
        (iso_z(parse_ts(start)), iso_z(parse_ts(end)), *SENSOR_METRICS),
    )
    conn.commit()
    return cur.rowcount


def ingest_lines(
    conn: sqlite3.Connection,
    lines: Iterable[str],
    start: str | datetime,
    cfg: TraysConfig,
    source: str = "simulated",
) -> int:
    """Ingest hub lines whose millis are relative to `start` (the simulated hub has no host clock)."""
    t0 = parse_ts(start)

    def samples() -> Iterable[Sample]:
        for line in lines:
            p = parse_line(line)
            if p:
                yield t0 + timedelta(milliseconds=p[0]), p[1], p[2], p[3]

    return insert_samples(conn, samples(), cfg, source)


def simulate_into_db(
    conn: sqlite3.Connection,
    cfg: TraysConfig,
    sim_cfg,  # backend.sim.sensor_hub.SimConfig
    start: str | datetime = "2015-12-14T00:00:00Z",
    end: str | datetime = "2016-01-18T23:59:00Z",
    replace: bool = False,
) -> int:
    """Run the simulated hub over [start, end] and ingest it (default: the whole dataset span).

    replace=True first deletes the simulated/derived rows in the range, so a changed seed or config
    takes effect (INSERT OR IGNORE alone would keep the stale rows).
    """
    from backend.sim.sensor_hub import SimulatedHub

    if replace:
        clear_simulated(conn, start, end)

    hub = SimulatedHub(cfg, sim_cfg.seed, sim_cfg.step_s, start=start, end=end, injections=sim_cfg.injections)
    return ingest_lines(conn, hub.lines(), start, cfg, source="simulated")


def read_serial(conn: sqlite3.Connection, cfg: TraysConfig, port: str, baud: int = 115200) -> None:
    """Optional real-hardware path: stamp each line with host UTC receipt time. Needs pyserial.

    Out of scope under PLAN 0.2 (no hardware), so untested: it writes derived rows per line and the
    sensor-hub CSV has no batching of one sample set per timestamp.
    """
    import serial  # type: ignore[import-not-found]

    with serial.Serial(port, baud, timeout=1) as ser:
        while True:
            raw = ser.readline().decode("ascii", errors="ignore")
            p = parse_line(raw)
            if p:
                insert_samples(conn, [(datetime.now(UTC), p[1], p[2], p[3])], cfg, "sensor")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest sensor-hub CSV into sensor_readings.")
    ap.add_argument("--db", default="data/ceres.db")
    ap.add_argument("--csv", help="CSV file of hub lines (use with --start)")
    ap.add_argument("--start", help="ISO start time that millis are relative to")
    ap.add_argument("--serial", help="serial port of a real hub (needs pyserial)")
    args = ap.parse_args(argv)
    conn = init_db(args.db)
    cfg = load_trays()
    if args.serial:
        read_serial(conn, cfg, args.serial)
    elif args.csv and args.start:
        with open(args.csv, encoding="utf-8") as f:
            print(ingest_lines(conn, f, args.start, cfg), "rows inserted")
    else:
        ap.error("give --serial PORT, or --csv FILE --start ISO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
