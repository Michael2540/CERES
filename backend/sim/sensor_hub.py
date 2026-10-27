"""Simulated sensor hub (PLAN.md 0.2).

A seeded, deterministic diurnal model that emits the hub CSV protocol, one line per reading:
"<millis>,<tray_or_air>,<metric>,<value>" (millis since the simulation start). The hub accepts the
same commands as the real one: "HOLD <tray>:<metric>", "RELEASE <tray>:<metric>" (a bare metric means
tray "air") and "PING". While a channel is held its last value is repeated, which produces a real
flatline for the SENSOR_FAULT story. All readings from this module carry source="simulated".
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from pydantic import BaseModel

from backend.config import ROOT, TraysConfig, load_trays
from backend.timeutil import lamp_is_on, parse_ts

SIM_YAML = ROOT / "config" / "sim.yaml"
WATER_EPOCH = "2015-12-14T09:00:00Z"  # first watering; then every 48 h (09:00 UTC)
WATER_EVERY_H = 48
WATER_TARGET = 0.9  # wetness fraction the soil is restored toward (1 = wet_raw, 0 = dry_raw)
WATER_GAIN = 0.9  # share of the gap to WATER_TARGET closed by one watering
DRAW_LAMP_ON, DRAW_LAMP_OFF = 0.006, 0.0015  # wetness fraction lost per hour, lamp on / off
NO_WATER_STRESS = 3.5  # lamp-on drawdown multiplier for the withheld-water tray after fault_start
SOIL_INIT, SOIL_FLOOR = 0.8, 0.05
STATE_EPOCH = "2015-12-14T00:00:00Z"  # dataset start: soil state is always warmed up from here
DOME_RAMP_H = 6.0


class Injection(BaseModel):
    at: str
    command: str
    duration_min: float | None = None  # None: never released


class SimConfig(BaseModel):
    seed: int = 0
    step_s: int = 300
    injections: list[Injection] = []


def load_sim_config(path: str | Path = SIM_YAML) -> SimConfig:
    with open(path, encoding="utf-8") as f:
        return SimConfig.model_validate(yaml.safe_load(f))


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def lamp_fraction(dt: datetime, on_h: float, off_h: float) -> float:
    """Smooth 0..1 lamp influence on the air: 1 h ramp up after lamp-on, 1 h ramp down after lamp-off."""
    h = dt.hour + dt.minute / 60 + dt.second / 3600
    if on_h <= h < off_h:
        return _smooth(h - on_h)
    return 1 - _smooth((h - off_h) % 24)


class SimulatedHub:
    def __init__(
        self,
        cfg: TraysConfig,
        seed: int = 0,
        step_s: int = 300,
        *,
        start: str | datetime,
        end: str | datetime,
        injections: Sequence[Injection] = (),
    ) -> None:
        self.cfg, self.seed, self.step_s = cfg, seed, step_s
        self.start, self.end = parse_ts(start), parse_ts(end)
        self.injections = list(injections)
        self._held: set[tuple[str, str]] = set()
        self._last: dict[tuple[str, str], float] = {}
        lamp = cfg.trays[0].lamp
        self._on_h = int(lamp.on[:2]) + int(lamp.on[3:]) / 60
        self._off_h = int(lamp.off[:2]) + int(lamp.off[3:]) / 60
        self._soil = {t.id: SOIL_INIT for t in cfg.trays}
        self._epoch = parse_ts(WATER_EPOCH)
        self._fs = {t.id: parse_ts(t.fault_start) if t.fault_start else None for t in cfg.trays}
        self._state_epoch = parse_ts(STATE_EPOCH)

    def command(self, cmd: str) -> str:
        """Handle a hub command and return the response line."""
        parts = cmd.strip().split()
        if len(parts) == 1 and parts[0].upper() == "PING":
            return "PONG"
        if len(parts) == 2 and parts[0].upper() in ("HOLD", "RELEASE"):
            tray, _, metric = parts[1].rpartition(":")
            key = (tray or "air", metric)
            if parts[0].upper() == "HOLD":
                self._held.add(key)
            else:
                self._held.discard(key)
                self._last.pop(key, None)
            return "OK"
        return "ERR unknown command"

    def _events(self) -> list[tuple[datetime, str]]:
        ev: list[tuple[datetime, str]] = []
        for inj in self.injections:
            at = parse_ts(inj.at)
            ev.append((at, inj.command))
            if inj.duration_min is not None and inj.command.upper().startswith("HOLD"):
                target = inj.command.split(None, 1)[1]
                ev.append((at + timedelta(minutes=inj.duration_min), f"RELEASE {target}"))
        return sorted(ev, key=lambda e: e[0])

    def _watered(self, t: datetime) -> bool:
        """True if a watering event falls in (t - step, t]."""
        k = math.floor((t - self._epoch) / timedelta(hours=WATER_EVERY_H))
        ev = self._epoch + k * timedelta(hours=WATER_EVERY_H)
        return k >= 0 and t - timedelta(seconds=self.step_s) < ev <= t

    def _advance(self, t: datetime) -> None:
        """Advance every tray's soil wetness by one step to time t (drawdown, then watering)."""
        dt_h = self.step_s / 3600
        for tray in self.cfg.trays:
            fs = self._fs[tray.id]
            withheld = tray.treatment == "no_water" and fs is not None and t >= fs
            lamp_on = lamp_is_on(tray.lamp, t)
            rate = (DRAW_LAMP_ON * (NO_WATER_STRESS if withheld else 1.0)) if lamp_on else DRAW_LAMP_OFF
            s = self._soil[tray.id] - rate * dt_h
            if self._watered(t) and not withheld:
                s += WATER_GAIN * (WATER_TARGET - s)
            self._soil[tray.id] = min(1.0, max(SOIL_FLOOR, s))

    def _sample(self, t: datetime, k: int) -> Iterator[tuple[str, str, float]]:
        rng = random.Random(self.seed * 1_000_003 + k)  # fixed draw order: holds never shift the stream
        lf = lamp_fraction(t, self._on_h, self._off_h)
        temp = 19.0 + 4.0 * lf + rng.gauss(0, 0.08)
        rh = 60.0 - (temp - 21.0) * 2.5 + rng.gauss(0, 0.4)
        co2 = 950.0 - 130.0 * lf + rng.gauss(0, 8)
        yield "air", "temp_c", round(temp, 2)
        yield "air", "rh_pct", round(rh, 2)
        yield "air", "co2_ppm", round(co2, 1)
        self._advance(t)
        for tray in self.cfg.trays:
            fs = self._fs[tray.id]
            s = self._soil[tray.id]
            raw = tray.soil.dry_raw - s * (tray.soil.dry_raw - tray.soil.wet_raw) + rng.gauss(0, 4)
            yield tray.id, "soil_raw", float(round(raw))
            if tray.treatment == "humid_no_fan":
                ramp = _smooth((t - fs) / timedelta(hours=DOME_RAMP_H)) if fs and t >= fs else 0.0
                target = tray.fault.rh_target_pct if tray.fault.rh_target_pct is not None else rh
                rh_dome = rh + (target - rh) * ramp + rng.gauss(0, 0.3)
                yield tray.id, "temp_c", round(temp + 0.5 + rng.gauss(0, 0.05), 2)
                yield tray.id, "rh_pct", round(min(99.0, rh_dome), 2)

    def lines(self) -> Iterator[str]:
        """Yield CSV lines for every grid step in [start, end], applying scheduled injections.

        The grid is anchored at 2015-12-14T00:00Z (multiples of step_s), and soil state is warmed up from
        there, so a given timestamp has the same values whatever start/end are requested (same step_s).
        Calling lines() again restarts from a clean state.
        """
        self._held.clear()
        self._last.clear()
        self._soil = {t.id: SOIL_INIT for t in self.cfg.trays}
        step = timedelta(seconds=self.step_s)
        k0 = -((self._state_epoch - self.start) // step)  # first grid index at or after start
        for k in range(k0):
            self._advance(self._state_epoch + k * step)
        events = self._events()
        ev_i = 0
        k = k0
        while (t := self._state_epoch + k * step) <= self.end:
            while ev_i < len(events) and events[ev_i][0] <= t:
                self.command(events[ev_i][1])
                ev_i += 1
            millis = round((t - self.start).total_seconds() * 1000)
            for tray, metric, value in self._sample(t, k):
                key = (tray, metric)
                if key in self._held:
                    value = self._last.setdefault(key, value)
                else:
                    self._last[key] = value
                yield f"{millis},{tray},{metric},{value:g}"
            k += 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Simulated sensor hub: write CSV lines to stdout.")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--sim-config", default=str(SIM_YAML))
    ap.add_argument("--trays-config", default=None)
    args = ap.parse_args(argv)
    sim = load_sim_config(args.sim_config)
    cfg = load_trays(args.trays_config) if args.trays_config else load_trays()
    hub = SimulatedHub(cfg, sim.seed, sim.step_s, start=args.start, end=args.end, injections=sim.injections)
    for line in hub.lines():
        sys.stdout.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
