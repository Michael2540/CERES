"""Settings from the environment (.env via python-dotenv) and typed tray config from config/trays.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
TRAYS_YAML = ROOT / "config" / "trays.yaml"

EDGE_MODES = ("recorded", "remote_glimmer", "spark_contributor")


def _bool(v: str | None) -> bool:
    return (v or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    db_path: str
    sam_backend: str
    edge_mode: str
    budget_usd: float
    demo_mode: bool
    has_meta_api_key: bool  # presence only; the value is never stored or logged


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    edge = os.getenv("EDGE_MODE") or "recorded"
    if edge not in EDGE_MODES:
        raise ValueError(f"EDGE_MODE must be one of {EDGE_MODES}, got {edge!r}")
    return Settings(
        db_path=os.getenv("DB_PATH") or "data/ceres.db",
        sam_backend=os.getenv("SAM_BACKEND") or "api",
        edge_mode=edge,
        budget_usd=float(os.getenv("CERES_BUDGET_USD") or 100),
        demo_mode=_bool(os.getenv("DEMO_MODE")),
        has_meta_api_key=bool(os.getenv("META_API_KEY")),
    )


class Lamp(BaseModel):
    on: str  # "HH:MM" UTC
    off: str


class SoilCalibration(BaseModel):
    dry_raw: float
    wet_raw: float


class FaultParams(BaseModel):
    erode_px_per_day: float | None = None  # tray B
    desat_per_day: float | None = None  # tray B
    growth_factor: float | None = None  # tray C
    brightness_factor: float | None = None  # tray C
    rh_target_pct: float | None = None  # tray L


class TrayConfig(BaseModel):
    id: str
    crop: str
    aled_tray: str
    treatment: Literal["control", "no_water", "low_light", "humid_no_fan"]
    fault_start: str | None = None
    reference_tray_id: str | None = None
    roi: tuple[int, int, int, int]
    cm2_per_px: float
    calib_flag: str
    lamp: Lamp
    ppfd_est: float
    soil: SoilCalibration
    fault: FaultParams = FaultParams()


class TraysConfig(BaseModel):
    trays: list[TrayConfig]

    def by_id(self, tray_id: str) -> TrayConfig:
        return next(t for t in self.trays if t.id == tray_id)


def load_trays(path: str | Path = TRAYS_YAML) -> TraysConfig:
    with open(path, encoding="utf-8") as f:
        return TraysConfig.model_validate(yaml.safe_load(f))
