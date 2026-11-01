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
    images_root: str = ""  # IMAGES_ROOT: directory holding the frame files by name (the demo bundle); "" = repo root
    replay_pace_s: float = 0.6  # CERES_REPLAY_PACE_S: pause between replayed events so the SSE stream is readable


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    edge = os.getenv("EDGE_MODE") or "recorded"
    if edge not in EDGE_MODES:
        raise ValueError(f"EDGE_MODE must be one of {EDGE_MODES}, got {edge!r}")
    demo = _bool(os.getenv("DEMO_MODE"))
    if demo:
        edge = "recorded"  # DEMO_MODE never makes a live call, whatever EDGE_MODE says
    return Settings(
        db_path=os.getenv("DB_PATH") or "data/ceres.db",
        sam_backend=os.getenv("SAM_BACKEND") or "api",
        edge_mode=edge,
        budget_usd=float(os.getenv("CERES_BUDGET_USD") or 100),
        demo_mode=demo,
        has_meta_api_key=bool(os.getenv("META_API_KEY")),
        images_root=os.getenv("IMAGES_ROOT") or "",
        replay_pace_s=float(os.getenv("CERES_REPLAY_PACE_S") or 0.6),
    )


def frame_file(path: str | Path) -> Path:
    """Where a frame's image lives. `frames.path` is relative to the repo root; with IMAGES_ROOT set (the offline demo)
    the file is looked up by name in that directory instead, so the demo serves data/demo/frames without data/raw."""
    root = load_settings().images_root
    if root:
        base = Path(root)
        return (base if base.is_absolute() else ROOT / base) / Path(str(path)).name
    return ROOT / path


class Lamp(BaseModel):
    on: str  # "HH:MM" UTC
    off: str


class SoilCalibration(BaseModel):
    dry_raw: float
    wet_raw: float


class FaultParams(BaseModel):
    shrink_per_day: float | None = None  # tray B: rosette scale loss per day, s(d) = max(s_min, 1 - shrink*d)
    s_min: float | None = None  # tray B: smallest rosette scale
    desat_per_day: float | None = None  # tray B: colour dulling per day
    growth_factor: float | None = None  # tray C
    brightness_factor: float | None = None  # tray C
    rh_target_pct: float | None = None  # tray L


class PotGrid(BaseModel):
    """The pots of a tray: an even cols x rows split of the tray ROI, with optional per-cell bounds
    (keys "r,c", 0-based; values x,y,w,h px) overriding the even split where it is off."""

    cols: int = 5
    rows: int = 4
    cells: dict[str, tuple[int, int, int, int]] = {}


class TrayConfig(BaseModel):
    id: str
    crop: str
    hardware: str = ""  # physical layout shown to the agent (never the treatment or the fault)
    aled_tray: str
    treatment: Literal["control", "no_water", "low_light", "humid_no_fan"]
    fault_start: str | None = None
    reference_tray_id: str | None = None
    roi: tuple[int, int, int, int]
    roi_margin_px: int = 40  # canopy is clipped to the ROI grown by this margin (clamped to the frame)
    pot_grid: PotGrid = PotGrid()
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
