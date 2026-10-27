"""Tests for the simulated sensor hub (backend/sim/sensor_hub.py)."""

from __future__ import annotations

import re

from backend.config import load_trays
from backend.science.vpd import vpd_kpa
from backend.sim.sensor_hub import SimulatedHub, load_sim_config
from backend.timeutil import parse_ts

CFG = load_trays()
LINE = re.compile(r"^\d+,(air|A|B|C|L),[a-z0-9_]+,-?\d+(\.\d+)?$")


def hub(start: str, end: str, seed: int = 0, step_s: int = 300) -> SimulatedHub:
    return SimulatedHub(CFG, seed, step_s, start=start, end=end)


def parsed(lines: list[str]) -> dict[tuple[str, str], list[tuple[int, float]]]:
    out: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for ln in lines:
        ms, tray, metric, val = ln.split(",")
        out.setdefault((tray, metric), []).append((int(ms), float(val)))
    return out


def test_deterministic_by_seed() -> None:
    a = list(hub("2016-01-02T00:00:00Z", "2016-01-02T06:00:00Z").lines())
    assert a == list(hub("2016-01-02T00:00:00Z", "2016-01-02T06:00:00Z").lines())
    assert a != list(hub("2016-01-02T00:00:00Z", "2016-01-02T06:00:00Z", seed=1).lines())


def test_csv_format_and_millis() -> None:
    lines = list(hub("2016-01-02T00:00:00Z", "2016-01-02T01:00:00Z").lines())
    assert all(LINE.match(ln) for ln in lines)
    series = parsed(lines)
    assert [m for m, _ in series[("air", "temp_c")]] == [i * 300_000 for i in range(13)]
    assert {k for k in series} >= {("air", "temp_c"), ("air", "rh_pct"), ("air", "co2_ppm"), ("B", "soil_raw"), ("L", "rh_pct")}


def test_diurnal_air() -> None:
    s = parsed(list(hub("2016-01-02T00:00:00Z", "2016-01-02T23:55:00Z").lines()))
    temp = dict(s[("air", "temp_c")])
    h = 3_600_000
    assert 22.5 < temp[14 * h] < 23.5 and 18.5 < temp[4 * h] < 19.5
    assert dict(s[("air", "co2_ppm")])[14 * h] < dict(s[("air", "co2_ppm")])[4 * h]
    assert dict(s[("air", "rh_pct")])[14 * h] < dict(s[("air", "rh_pct")])[4 * h]


def test_ping_and_commands() -> None:
    h = hub("2016-01-02T00:00:00Z", "2016-01-02T01:00:00Z")
    assert h.command("PING") == "PONG"
    assert h.command("HOLD air:rh_pct") == "OK" and h._held == {("air", "rh_pct")}
    assert h.command("RELEASE rh_pct") == "OK" and not h._held  # bare metric means tray air
    assert h.command("HOLD rh_pct") == "OK" and h._held == {("air", "rh_pct")}
    assert h.command("BOGUS").startswith("ERR")


def test_hold_repeats_then_release_resumes() -> None:
    h = hub("2016-01-02T10:00:00Z", "2016-01-02T12:00:00Z")
    lines: list[str] = []
    for ln in h.lines():
        ms = int(ln.split(",")[0])
        if ms == 1_800_000 and not h._held:
            h.command("HOLD air:rh_pct")
        if ms == 3_600_000 and h._held:
            h.command("RELEASE air:rh_pct")
        lines.append(ln)
    rh = dict(parsed(lines)[("air", "rh_pct")])
    held = {rh[ms] for ms in range(1_800_000, 3_600_000, 300_000)}
    assert held == {rh[1_500_000]}
    assert len({rh[ms] for ms in range(3_600_000, 4_800_000, 300_000)}) > 1
    temp = [v for _, v in parsed(lines)[("air", "temp_c")][6:12]]
    assert len(set(temp)) > 1  # other channels keep moving


def test_scheduled_injection_from_config() -> None:
    sim = load_sim_config()
    h = SimulatedHub(CFG, sim.seed, sim.step_s, start="2016-01-10T13:30:00Z", end="2016-01-10T15:30:00Z", injections=sim.injections)
    rh = dict(parsed(list(h.lines()))[("air", "rh_pct")])
    held = [rh[ms] for ms in range(1_800_000, 5_400_000, 300_000)]  # 14:00 .. 14:55
    assert len(set(held)) == 1 and rh[5_400_000] != held[0]  # released at 15:00 (duration_min: 60)


def test_b_dries_after_fault_start_while_a_is_watered() -> None:
    s = parsed(list(hub("2016-01-04T00:00:00Z", "2016-01-08T00:00:00Z", step_s=1800).lines()))
    t0 = parse_ts("2016-01-04T00:00:00Z")

    def at(key: tuple[str, str], iso: str) -> float:
        return dict(s[key])[int((parse_ts(iso) - t0).total_seconds() * 1000)]

    # 2016-01-05T09:00 is a watering time (every 48 h from 2015-12-14T09:00): A gets wetter, B does not.
    assert at(("A", "soil_raw"), "2016-01-05T11:00:00Z") < at(("A", "soil_raw"), "2016-01-05T07:00:00Z") - 200
    assert at(("B", "soil_raw"), "2016-01-05T11:00:00Z") > at(("B", "soil_raw"), "2016-01-05T07:00:00Z")
    b = [at(("B", "soil_raw"), f"2016-01-0{d}T12:00:00Z") for d in (5, 6, 7)]
    assert b[0] < b[1] < b[2]
    # A is watered again on 2016-01-07 and ends up wetter than B
    assert at(("A", "soil_raw"), "2016-01-07T12:00:00Z") < at(("B", "soil_raw"), "2016-01-07T12:00:00Z") - 400


def test_l_vpd_low_after_fault_start() -> None:
    s = parsed(list(hub("2016-01-07T00:00:00Z", "2016-01-09T00:00:00Z").lines()))
    t0, fs = parse_ts("2016-01-07T00:00:00Z"), parse_ts("2016-01-08T08:00:00Z")
    temp, rh, air_rh = dict(s[("L", "temp_c")]), dict(s[("L", "rh_pct")]), dict(s[("air", "rh_pct")])
    for ms, r in rh.items():
        t = t0.timestamp() + ms / 1000
        vpd = vpd_kpa(temp[ms], r)
        if t >= fs.timestamp() + 6 * 3600:
            assert vpd < 0.4
        if t < fs.timestamp():
            assert abs(r - air_rh[ms]) < 1.5  # dome tracks the air before the fault


def test_lines_twice_identical_and_resets_state() -> None:
    h = hub("2016-01-07T00:00:00Z", "2016-01-07T06:00:00Z")
    first = list(h.lines())
    h.command("HOLD air:rh_pct")  # a stray hold must not leak into a new run
    assert list(h.lines()) == first


def test_values_independent_of_start_and_end() -> None:
    def b_soil(start: str, end: str) -> float:
        s = parsed(list(hub(start, end).lines()))[("B", "soil_raw")]
        t0 = parse_ts(start)
        want = int((parse_ts("2016-01-05T12:00:00Z") - t0).total_seconds() * 1000)
        return dict(s)[want]

    a = b_soil("2016-01-05T00:00:00Z", "2016-01-05T13:00:00Z")
    assert a == b_soil("2016-01-03T07:00:00Z", "2016-01-06T00:00:00Z") == b_soil("2015-12-14T00:00:00Z", "2016-01-05T12:00:00Z")
