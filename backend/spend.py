"""Shared Meta budget ledger (PLAN 0.4b). Every paid call records a row; calls are refused past the cap."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from backend.config import load_settings

# USD prices. Tokens are per million; SAM is per image (SPIKES S1: usage.images_processed = 1).
PRICES: dict[str, dict[str, float]] = {
    "sam-3.1": {"image": 0.0025},
    "muse-spark-1.3": {"in_per_m": 1.25, "out_per_m": 4.25},
    "muse-spark-1.3-contributor": {"in_per_m": 0.10, "out_per_m": 0.20},
}


class BudgetExceeded(RuntimeError):
    """Raised before a call that would push the ledger past CERES_BUDGET_USD."""


def cost_for(model: str, input_tokens: int = 0, output_tokens: int = 0, images: int = 0) -> float:
    p = PRICES.get(model)
    if p is None:
        return 0.0  # local / fallback backends (e.g. hsv-fallback) are free
    return (
        images * p.get("image", 0.0)
        + input_tokens * p.get("in_per_m", 0.0) / 1e6
        + output_tokens * p.get("out_per_m", 0.0) / 1e6
    )


def record_spend(
    conn: sqlite3.Connection,
    role: str,
    provider: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    images: int = 0,
    cost_usd: float | None = None,
) -> float:
    """Insert a ledger row and return its cost (computed from PRICES when cost_usd is None)."""
    cost = cost_for(model, input_tokens, output_tokens, images) if cost_usd is None else cost_usd
    conn.execute(
        "INSERT INTO spend(ts, role, provider, model, input_tokens, output_tokens, images, cost_usd)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (datetime.now(UTC).isoformat(), role, provider, model, input_tokens, output_tokens, images, cost),
    )
    conn.commit()
    return cost


def total_spend(conn: sqlite3.Connection) -> float:
    return float(conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM spend").fetchone()[0])


def check_budget(conn: sqlite3.Connection, extra_usd: float = 0.0) -> None:
    cap = load_settings().budget_usd
    total = total_spend(conn)
    if total + extra_usd > cap:
        raise BudgetExceeded(f"budget ${cap:.2f} would be exceeded (spent ${total:.4f} + ${extra_usd:.4f})")
