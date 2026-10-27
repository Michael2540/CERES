import pytest

from backend.spend import BudgetExceeded, check_budget, cost_for, record_spend, total_spend


def test_prices():
    assert cost_for("sam-3.1", images=1000) == pytest.approx(2.5)
    assert cost_for("muse-spark-1.3", 1_000_000, 1_000_000) == pytest.approx(5.5)
    assert cost_for("muse-spark-1.3-contributor", 1_000_000, 1_000_000) == pytest.approx(0.3)
    assert cost_for("hsv-fallback", images=5) == 0.0


def test_ledger_and_guard(conn, monkeypatch):
    monkeypatch.setenv("CERES_BUDGET_USD", "0.01")
    record_spend(conn, "perception", "meta", "sam-3.1", images=2)
    assert total_spend(conn) == pytest.approx(0.005)
    check_budget(conn, 0.005)  # exactly at the cap is allowed
    with pytest.raises(BudgetExceeded):
        check_budget(conn, 0.0051)
