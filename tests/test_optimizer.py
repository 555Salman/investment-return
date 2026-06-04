"""
Unit tests for the portfolio optimizer.
Run with: python -m pytest tests/test_optimizer.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import numpy as np

from ml.optimization.portfolio_optimizer import (
    optimize_portfolio,
    equal_weight_portfolio,
    compare_strategies,
    INTEREST_RATES,
)

PAIRS = ["EUR_USD", "AUD_USD", "NZD_USD"]
MOCK_RETURNS = {"EUR_USD": 0.03, "AUD_USD": 0.04, "NZD_USD": 0.035}


class TestEqualWeight:

    def test_weights_sum_to_one(self):
        result = equal_weight_portfolio(MOCK_RETURNS)
        total = sum(result["allocations"].values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_each_weight_equals_one_third(self):
        result = equal_weight_portfolio(MOCK_RETURNS)
        for w in result["allocations"].values():
            assert w == pytest.approx(1 / 3, abs=1e-6)

    def test_amounts_sum_to_budget(self):
        budget = 5000.0
        result = equal_weight_portfolio(MOCK_RETURNS, budget=budget)
        assert sum(result["amounts"].values()) == pytest.approx(budget, abs=1e-4)

    def test_expected_amount_greater_than_budget(self):
        result = equal_weight_portfolio(MOCK_RETURNS, budget=10_000)
        assert result["expected_amount"] > result["budget"]


class TestOptimizePortfolio:

    def test_weights_sum_to_one(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "ml.optimization.portfolio_optimizer.get_risk_limits",
            lambda pairs, risk_tolerance: {p: 0.01 for p in pairs},
        )
        result = optimize_portfolio(MOCK_RETURNS, budget=10_000)
        total = sum(result["allocations"].values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_weights_within_bounds(self, monkeypatch):
        monkeypatch.setattr(
            "ml.optimization.portfolio_optimizer.get_risk_limits",
            lambda pairs, risk_tolerance: {p: 0.01 for p in pairs},
        )
        min_w, max_w = 0.05, 0.60
        result = optimize_portfolio(MOCK_RETURNS, min_weight=min_w, max_weight=max_w)
        for w in result["allocations"].values():
            assert w >= min_w - 1e-5
            assert w <= max_w + 1e-5

    def test_all_pairs_present(self, monkeypatch):
        monkeypatch.setattr(
            "ml.optimization.portfolio_optimizer.get_risk_limits",
            lambda pairs, risk_tolerance: {p: 0.01 for p in pairs},
        )
        result = optimize_portfolio(MOCK_RETURNS)
        assert set(result["allocations"].keys()) == set(MOCK_RETURNS.keys())

    def test_budget_allocation(self, monkeypatch):
        monkeypatch.setattr(
            "ml.optimization.portfolio_optimizer.get_risk_limits",
            lambda pairs, risk_tolerance: {p: 0.01 for p in pairs},
        )
        budget = 8000.0
        result = optimize_portfolio(MOCK_RETURNS, budget=budget)
        assert sum(result["amounts"].values()) == pytest.approx(budget, abs=1.0)


class TestInterestRates:

    def test_all_pairs_have_rates(self):
        for pair in PAIRS:
            assert pair in INTEREST_RATES
            assert 0 < INTEREST_RATES[pair] < 0.20   # sanity: between 0% and 20%
