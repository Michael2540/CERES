"""Deterministic alert rules (PLAN.md 7): SENSOR_FAULT, WILT_RISK, LOW_VPD_RISK, GROWTH_LAG.

Pure functions over plain lists of dicts, no I/O and no model calls:
  readings: [{id, ts, value, source}]       canopy: [{id, ts, area_px, exg, ...}]
Thresholds come from setpoints.yaml. Evidence ids name the rows that triggered a rule (kept few:
first / last / extreme points). Rules only look at rows with ts <= as_of. `evaluate` runs all rules for
one tray and suppresses plant alerts whose input sensor is itself faulted. Series are parsed and sorted
once (`Series`) and windows are found by bisection, so cost is linear in the rows actually used.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median
from typing import Any

from backend.config import TrayConfig
from backend.science.setpoints import Setpoints, default_setpoints
from backend.science.vpd import vpd_kpa
from backend.timeutil import iso_z, lamp_is_on, parse_ts

Row = dict[str, Any]

SEVERITY = {"SENSOR_FAULT": "warning", "WILT_RISK": "alert", "LOW_VPD_RISK": "watch", "GROWTH_LAG": "watch"}
FLAT_EPS = 1e-6
EVAL_WINDOW_H = 72  # evaluate() looks at this much trailing data (plus the baselines it needs)

# Episode gap: a rule that fires again within this many hours of its last firing continues the same alert
# episode. Plant rules use the visual signal only while the lamp is on, so they go quiet over the 10 h
# lamp-off night; 14 h keeps one fault as one episode. Sensor faults are instrument events: 2 h.
EPISODE_GAP_H = {"SENSOR_FAULT": 2}
PLANT_EPISODE_GAP_H = 14


def episode_gap_h(rule: str) -> int:
    return EPISODE_GAP_H.get(rule, PLANT_EPISODE_GAP_H)
GROWTH_CONFIRM_H = 12  # GROWTH_LAG must also have held this long before as_of (see growth_lag)


@dataclass(frozen=True)
class Alert:
    rule: str
    severity: str
    tray_id: str
    ts: str
    evidence_ids: tuple[str, ...]
    threshold_tag: str
    detail: str = ""


@dataclass(frozen=True)
class Series:
    """Rows parsed and sorted by time once; windows are bisected."""

    ts: list[datetime]
    rows: list[Row]

    @classmethod
    def of(cls, rows: Sequence[Row] | Series) -> Series:
        if isinstance(rows, Series):
            return rows
        pairs = sorted(((parse_ts(r["ts"]), r) for r in rows), key=lambda p: p[0])
        return cls([p[0] for p in pairs], [p[1] for p in pairs])

    def __len__(self) -> int:
        return len(self.rows)

    def slice(self, i: int, j: int) -> Series:
        return Series(self.ts[i:j], self.rows[i:j])

    def upto(self, t: datetime) -> Series:
        """Rows with ts <= t."""
        return self.slice(0, bisect_right(self.ts, t))

    def between(self, lo: datetime, hi: datetime) -> Series:
        """Rows with lo <= ts <= hi."""
        return self.slice(bisect_left(self.ts, lo), bisect_right(self.ts, hi))


def _alert(rule: str, tray_id: str, as_of: datetime, ids: Sequence[str], tag: str, detail: str = "") -> Alert:
    return Alert(rule, SEVERITY[rule], tray_id, iso_z(as_of), tuple(dict.fromkeys(ids)), tag, detail)


def _lamp_at(lamp: Series, tray: TrayConfig, ts: datetime) -> bool:
    """Lamp state at ts: latest lamp_on reading at or before ts, else the tray schedule."""
    i = bisect_right(lamp.ts, ts) - 1
    return lamp.rows[i]["value"] > 0.5 if i >= 0 else lamp_is_on(tray.lamp, ts)


# --- SENSOR_FAULT ------------------------------------------------------------------------------------
def _flat_runs(s: Series, min_dur: timedelta) -> list[tuple[int, int]]:
    """Index ranges (i, j) of maximal runs of constant value lasting at least min_dur (>= 3 points)."""
    runs, i, n = [], 0, len(s)
    while i < n:
        j = i
        while j + 1 < n and abs(s.rows[j + 1]["value"] - s.rows[i]["value"]) <= FLAT_EPS:
            j += 1
        if j - i >= 2 and s.ts[j] - s.ts[i] >= min_dur:
            runs.append((i, j))
        i = j + 1
    return runs


def sensor_fault(
    tray_id: str,
    metric: str,
    readings: Sequence[Row] | Series,
    as_of: str | datetime,
    sp: Setpoints | None = None,
) -> Alert | None:
    """A flatline (constant for >= flatline_min), gap (> gap_min without data) or out-of-limits value
    that is ongoing or ended within the last fault_lookback_min is reported (latched). Evidence is the
    first and last row of the run; `detail` carries its start and end."""
    sp, as_of = sp or default_setpoints(), parse_ts(as_of)
    s = Series.of(readings).upto(as_of)
    if not len(s):
        return None
    look = as_of - timedelta(minutes=sp.value("fault_lookback_min"))
    tag = f"sensor_fault:{sp.tag('flatline_min')}"
    recent = s.between(look, as_of)
    if (lim := sp.limit(metric)) is not None:
        bad = [i for i, r in enumerate(recent.rows) if not lim[0] <= r["value"] <= lim[1]]
        if bad:
            a, b = bad[0], bad[-1]
            span = f"{iso_z(recent.ts[a])}..{iso_z(recent.ts[b])}"
            ids = [recent.rows[a]["id"], recent.rows[b]["id"]]
            return _alert("SENSOR_FAULT", tray_id, as_of, ids, tag, f"{metric} out of range {span}")
    flat = timedelta(minutes=sp.value("flatline_min"))
    runs = [r for r in _flat_runs(s, flat) if s.ts[r[1]] >= look]
    if runs:
        i, j = runs[-1]
        span = f"{iso_z(s.ts[i])}..{iso_z(s.ts[j])}"
        return _alert("SENSOR_FAULT", tray_id, as_of, [s.rows[i]["id"], s.rows[j]["id"]], tag, f"{metric} flatline {span}")
    gap = timedelta(minutes=sp.value("gap_min"))
    if as_of - s.ts[-1] > gap:
        span = f"{iso_z(s.ts[-1])}..{iso_z(as_of)}"
        return _alert("SENSOR_FAULT", tray_id, as_of, [s.rows[-1]["id"]], tag, f"{metric} gap {span}")
    for k in range(max(bisect_left(s.ts, look) - 1, 0), len(s) - 1):
        if s.ts[k + 1] >= look and s.ts[k + 1] - s.ts[k] > gap:
            span = f"{iso_z(s.ts[k])}..{iso_z(s.ts[k + 1])}"
            ids = [s.rows[k]["id"], s.rows[k + 1]["id"]]
            return _alert("SENSOR_FAULT", tray_id, as_of, ids, tag, f"{metric} gap {span}")
    return None


# --- WILT_RISK ---------------------------------------------------------------------------------------
def baseline_window(tray: TrayConfig, soil_vwc: Sequence[Row] | Series) -> tuple[datetime, datetime] | None:
    """48 h before fault_start, or the first 48 h of data for a tray without a fault."""
    if tray.fault_start:
        end = parse_ts(tray.fault_start)
        return end - timedelta(hours=48), end
    s = Series.of(soil_vwc)
    return (s.ts[0], s.ts[0] + timedelta(hours=48)) if len(s) else None


def _valid_values(s: Series, metric: str, sp: Setpoints) -> list[float]:
    """Values of s that are inside the physical limits and not part of a flatline run."""
    lim = sp.limit(metric)
    skip = {k for i, j in _flat_runs(s, timedelta(minutes=sp.value("flatline_min"))) for k in range(i, j + 1)}
    return [r["value"] for k, r in enumerate(s.rows) if k not in skip and (lim is None or lim[0] <= r["value"] <= lim[1])]


def baseline_value(
    tray: TrayConfig,
    soil_vwc: Sequence[Row] | Series,
    as_of: str | datetime,
    window: tuple[datetime, datetime] | None = None,
    sp: Setpoints | None = None,
) -> float | None:
    """Median valid soil_vwc over the baseline window, using only data with ts <= as_of."""
    sp = sp or default_setpoints()
    s = Series.of(soil_vwc).upto(parse_ts(as_of))
    window = window or baseline_window(tray, s)
    if window is None:
        return None
    w = s.slice(bisect_left(s.ts, window[0]), bisect_left(s.ts, window[1]))
    vals = _valid_values(w, "soil_vwc", sp)
    return median(vals) if vals else None


def wilt_risk(
    tray: TrayConfig,
    soil_vwc: Sequence[Row] | Series,
    canopy: Sequence[Row] | Series,
    lamp_on: Sequence[Row] | Series,
    as_of: str | datetime,
    baseline: tuple[datetime, datetime] | None = None,
    sp: Setpoints | None = None,
    base_value: float | None = None,
) -> Alert | None:
    """Soil_vwc below wilt_soil_frac x baseline median AND, within wilt_window_h with the lamp on,
    canopy area down > wilt_area_drop OR ExG down > wilt_exg_drop (peak to latest)."""
    sp, as_of = sp or default_setpoints(), parse_ts(as_of)
    soil = Series.of(soil_vwc).upto(as_of)
    if not len(soil):
        return None
    base = base_value if base_value is not None else baseline_value(tray, soil, as_of, baseline, sp)
    last_soil = soil.rows[-1]
    if base is None or not last_soil["value"] < sp.value("wilt_soil_frac") * base:
        return None
    lamp = Series.of(lamp_on)
    win = Series.of(canopy).between(as_of - timedelta(hours=sp.value("wilt_window_h")), as_of)
    rows = [r for t, r in zip(win.ts, win.rows, strict=True) if _lamp_at(lamp, tray, t)]
    if len(rows) < 2:
        return None
    last = rows[-1]
    for key, thr in (("area_px", "wilt_area_drop"), ("exg", "wilt_exg_drop")):
        vals = [r for r in rows if r.get(key) is not None]
        if len(vals) < 2 or vals[-1] is not last:
            continue
        peak = max(vals, key=lambda r: r[key])
        if peak[key] > 0 and (peak[key] - last[key]) / peak[key] > sp.value(thr):
            ids = [last_soil["id"], peak["id"], last["id"]]
            return _alert("WILT_RISK", tray.id, as_of, ids, sp.tag("wilt_soil_frac"), f"soil + {key}")
    return None


# --- LOW_VPD_RISK ------------------------------------------------------------------------------------
def low_vpd_risk(
    tray: TrayConfig,
    vpd_series: Sequence[Row] | Series,
    lamp_on: Sequence[Row] | Series,
    as_of: str | datetime,
    sp: Setpoints | None = None,
) -> Alert | None:
    """VPD < vpd_low in the trailing contiguous run, with more than low_vpd_hours of lamp-on time in it."""
    sp, as_of = sp or default_setpoints(), parse_ts(as_of)
    s, lamp = Series.of(vpd_series).upto(as_of), Series.of(lamp_on)
    low, tol = sp.value("vpd_low"), timedelta(minutes=2 * sp.value("gap_min"))
    i = len(s)
    while i > 0 and s.rows[i - 1]["value"] < low and (i == len(s) or s.ts[i] - s.ts[i - 1] <= tol):
        i -= 1
    run = s.slice(i, len(s))
    if len(run) < 2:
        return None
    on = sum((run.ts[k + 1] - run.ts[k] for k in range(len(run) - 1) if _lamp_at(lamp, tray, run.ts[k])), timedelta())
    if on <= timedelta(hours=sp.value("low_vpd_hours")):
        return None
    lowest = min(run.rows, key=lambda r: r["value"])
    ids = [run.rows[0]["id"], lowest["id"], run.rows[-1]["id"]]
    return _alert("LOW_VPD_RISK", tray.id, as_of, ids, sp.tag("vpd_low"), f"VPD < {low} kPa")


def vpd_series(temp: Sequence[Row] | Series, rh: Sequence[Row] | Series) -> list[Row]:
    """Air VPD rows from temp/RH readings with identical timestamps; id = the RH reading id."""
    t_by_ts = {r["ts"]: r["value"] for r in Series.of(temp).rows}
    return [
        {"id": r["id"], "ts": r["ts"], "value": vpd_kpa(t_by_ts[r["ts"]], r["value"]), "source": "derived"}
        for r in Series.of(rh).rows
        if r["ts"] in t_by_ts
    ]


# --- GROWTH_LAG --------------------------------------------------------------------------------------
def _growth(rows: Series, as_of: datetime, window_h: float) -> tuple[float, Row, Row] | None:
    """Relative canopy growth per `window_h` hours over the window ending at the latest row <= as_of."""
    up = rows.upto(as_of)
    keep = [k for k, r in enumerate(up.rows) if r.get("area_px")]
    pts = Series([up.ts[k] for k in keep], [up.rows[k] for k in keep])
    if not len(pts):
        return None
    t1, end = pts.ts[-1], pts.rows[-1]
    i = bisect_right(pts.ts, t1 - timedelta(hours=window_h)) - 1
    if i < 0:
        return None
    t0, start = pts.ts[i], pts.rows[i]
    span_h = (t1 - t0).total_seconds() / 3600
    if span_h > 1.5 * window_h:
        return None
    return (end["area_px"] / start["area_px"] - 1) * window_h / span_h, start, end


def own_growth_baseline(
    tray: TrayConfig,
    canopy: Sequence[Row] | Series,
    as_of: str | datetime | None = None,
    sp: Setpoints | None = None,
) -> float | None:
    """The tray's own growth rate over the window ending at fault_start (None without a fault).
    With as_of given, only data up to as_of is used."""
    sp = sp or default_setpoints()
    if not tray.fault_start:
        return None
    end = parse_ts(tray.fault_start)
    if as_of is not None:
        end = min(end, parse_ts(as_of))
    g = _growth(Series.of(canopy), end, sp.value("growth_window_h"))
    return g[0] if g else None


def growth_lag(
    tray: TrayConfig,
    canopy: Sequence[Row] | Series,
    ref_canopy: Sequence[Row] | Series | None,
    as_of: str | datetime,
    own_baseline: float | None = None,
    sp: Setpoints | None = None,
) -> Alert | None:
    """Growth over growth_window_h (24 h) below growth_lag_frac x the reference's (or own pre-fault) rate.

    Demo-time adaptation of "for 24 h": the condition must hold at as_of AND at as_of - 12 h (each over its
    own 24 h window), not for a further 24 h. A tray with reference_tray_id but no reference canopy is not
    evaluated (None); there is no silent fallback to the own-baseline.
    """
    sp, as_of = sp or default_setpoints(), parse_ts(as_of)
    win, frac = sp.value("growth_window_h"), sp.value("growth_lag_frac")
    own, ref = Series.of(canopy), None
    if tray.reference_tray_id:
        if ref_canopy is None or not len(Series.of(ref_canopy)):
            return None
        ref = Series.of(ref_canopy)
    elif own_baseline is None:
        return None
    evidence: list[str] = []
    for k, t in enumerate((as_of, as_of - timedelta(hours=GROWTH_CONFIRM_H))):
        g = _growth(own, t, win)
        base_g = _growth(ref, t, win) if ref is not None else None
        if ref is not None and base_g is None:
            return None
        base = base_g[0] if base_g else own_baseline
        if g is None or base is None or not (base > 0 and g[0] < frac * base):
            return None
        if k == 0:
            evidence = [g[1]["id"], g[2]["id"]] + ([base_g[1]["id"], base_g[2]["id"]] if base_g else [])
    return _alert("GROWTH_LAG", tray.id, as_of, evidence, sp.tag("growth_lag_frac"), "growth below reference")


# --- evaluate ----------------------------------------------------------------------------------------
def evaluate(
    tray: TrayConfig,
    readings: Mapping[tuple[str, str], Sequence[Row]],
    canopy: Sequence[Row],
    ref_canopy: Sequence[Row] | None,
    as_of: str | datetime,
    sp: Setpoints | None = None,
    skipped: list[str] | None = None,
) -> list[Alert]:
    """Run every rule for one tray at as_of. `readings` is keyed by (tray_or_air, metric).

    Looks at the trailing EVAL_WINDOW_H hours before as_of plus the baselines (soil baseline, own
    growth baseline). Sensor faults are reported with tray_id = the sensor's owner ("air" for shared
    sensors, so callers merging several trays should de-duplicate on (rule, tray_id, detail)). Plant
    alerts that depend on a faulted sensor are suppressed: soil_raw -> WILT_RISK, climate temp_c/rh_pct
    -> LOW_VPD_RISK. Rules lacking inputs are named in `skipped` (if given).
    """
    sp, as_of = sp or default_setpoints(), parse_ts(as_of)
    trail = as_of - timedelta(hours=EVAL_WINDOW_H)
    skip = skipped if skipped is not None else []

    full: dict[tuple[str, str], Series] = {}

    def get_full(owner: str, metric: str) -> Series:
        if (owner, metric) not in full:
            full[(owner, metric)] = Series.of(readings.get((owner, metric)) or []).upto(as_of)
        return full[(owner, metric)]

    def get(owner: str, metric: str) -> Series:
        return get_full(owner, metric).between(trail, as_of)

    climate = tray.id if (tray.id, "rh_pct") in readings else "air"
    faults: dict[tuple[str, str], Alert] = {}
    for key in [(tray.id, "soil_raw"), (climate, "temp_c"), (climate, "rh_pct"), ("air", "co2_ppm")]:
        if (a := sensor_fault(key[0], key[1], get(*key), as_of, sp)) is not None:
            faults[key] = a
    out = list(faults.values())
    lamp = get(tray.id, "lamp_on")

    soil_full = get_full(tray.id, "soil_vwc")
    cano_full = Series.of(canopy).upto(as_of)
    cano = cano_full.between(trail, as_of)
    if (tray.id, "soil_raw") in faults:
        pass
    elif not len(cano) or not len(soil_full):
        skip.append("WILT_RISK")
    else:
        base = baseline_value(tray, soil_full, as_of, sp=sp)
        if a := wilt_risk(tray, soil_full.between(trail, as_of), cano, lamp, as_of, sp=sp, base_value=base):
            out.append(a)

    t_s, rh_s = get(climate, "temp_c"), get(climate, "rh_pct")
    if (climate, "temp_c") in faults or (climate, "rh_pct") in faults:
        pass
    elif not len(t_s) or not len(rh_s):
        skip.append("LOW_VPD_RISK")
    elif a := low_vpd_risk(tray, vpd_series(t_s, rh_s), lamp, as_of, sp):
        out.append(a)

    if not len(cano_full) or (tray.reference_tray_id and not ref_canopy):
        skip.append("GROWTH_LAG")
    else:
        ref = Series.of(ref_canopy).between(trail, as_of) if ref_canopy else None
        own = own_growth_baseline(tray, cano_full, as_of, sp)
        if a := growth_lag(tray, cano, ref, as_of, own, sp):
            out.append(a)
    return out
