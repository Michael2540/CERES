"""Vapour pressure deficit helpers (Tetens). All values are AIR VPD from air temperature and RH,
not leaf VPD (no leaf temperature is measured)."""

from __future__ import annotations

from math import exp


def svp_kpa(t_c: float) -> float:
    """Saturation vapour pressure in kPa at air temperature t_c (Tetens, over water)."""
    return 0.6108 * exp(17.27 * t_c / (t_c + 237.3))


def vpd_kpa(temp_c: float, rh_pct: float) -> float:
    """Air VPD in kPa: svp(T) * (1 - RH/100). 25 C / 50 % RH gives about 1.584 kPa."""
    return svp_kpa(temp_c) * (1.0 - rh_pct / 100.0)
