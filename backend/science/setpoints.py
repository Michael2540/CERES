"""Loader for backend/science/setpoints.yaml: rule thresholds, physical limits and provenance rows."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

SETPOINTS_YAML = Path(__file__).with_name("setpoints.yaml")


@dataclass(frozen=True)
class Setpoints:
    rows: list[dict[str, Any]]
    thresholds: dict[str, dict[str, Any]]
    limits: dict[str, dict[str, Any]]

    def value(self, name: str) -> float:
        return float(self.thresholds[name]["value"])

    def tag(self, name: str) -> str:
        return str(self.thresholds[name]["tag"])

    def limit(self, metric: str) -> tuple[float, float] | None:
        lim = self.limits.get(metric)
        return (float(lim["min"]), float(lim["max"])) if lim else None


def load_setpoints(path: str | Path = SETPOINTS_YAML) -> Setpoints:
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return Setpoints(d["setpoints"], d["thresholds"], d["physical_limits"])


@lru_cache(maxsize=1)
def default_setpoints() -> Setpoints:
    return load_setpoints()
