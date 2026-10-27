"""Tests for backend/science/vpd.py."""

from __future__ import annotations

import pytest

from backend.science.vpd import svp_kpa, vpd_kpa


def test_vpd_25c_50rh() -> None:
    assert vpd_kpa(25, 50) == pytest.approx(1.584, abs=0.01)


def test_vpd_saturated_air_is_zero() -> None:
    assert vpd_kpa(20, 100) == pytest.approx(0.0)


def test_svp_increases_with_temperature() -> None:
    assert svp_kpa(10) < svp_kpa(20) < svp_kpa(30)
    assert svp_kpa(20) == pytest.approx(2.34, abs=0.02)
