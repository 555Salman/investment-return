"""
Unit tests for evaluation metrics and backtesting utilities.
Run with: python -m pytest tests/test_backtest.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from ml.training.evaluate import (
    sharpe_ratio, max_drawdown, total_return, portfolio_metrics
)


class TestPortfolioMetrics:

    def test_total_return_positive(self):
        curve = np.array([100.0, 110.0, 121.0])
        assert total_return(curve) == pytest.approx(0.21, abs=1e-4)

    def test_total_return_negative(self):
        curve = np.array([100.0, 90.0, 80.0])
        assert total_return(curve) < 0

    def test_total_return_flat(self):
        curve = np.ones(10) * 100.0
        assert total_return(curve) == pytest.approx(0.0, abs=1e-6)

    def test_max_drawdown_no_decline(self):
        curve = np.array([100.0, 110.0, 120.0, 130.0])
        assert max_drawdown(curve) == pytest.approx(0.0, abs=1e-6)

    def test_max_drawdown_known_value(self):
        # peaks at 120, drops to 90 → drawdown = (90-120)/120 = -0.25
        curve = np.array([100.0, 120.0, 90.0, 100.0])
        assert max_drawdown(curve) == pytest.approx(-0.25, abs=1e-4)

    def test_sharpe_flat_returns_zero(self):
        # Constant equity → zero std → sharpe = 0
        # Constant returns → std = 0 → sharpe guard returns 0.0
        returns = np.full(251, 0.001)
        result  = sharpe_ratio(returns)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_portfolio_metrics_keys(self):
        curve   = np.linspace(100, 115, 100)
        metrics = portfolio_metrics(curve)
        expected_keys = {
            "total_return_pct", "sharpe_ratio", "max_drawdown_pct",
            "annualised_vol_pct", "final_value", "initial_value"
        }
        assert expected_keys <= set(metrics.keys())

    def test_portfolio_metrics_final_value(self):
        curve   = np.linspace(10_000, 11_000, 50)
        metrics = portfolio_metrics(curve)
        assert metrics["final_value"] == pytest.approx(11_000.0, abs=1.0)
        assert metrics["initial_value"] == pytest.approx(10_000.0, abs=1.0)

    def test_growing_portfolio_positive_return(self):
        curve   = np.linspace(10_000, 12_000, 100)
        metrics = portfolio_metrics(curve)
        assert metrics["total_return_pct"] > 0
        assert metrics["max_drawdown_pct"] == pytest.approx(0.0, abs=1e-3)

    def test_declining_portfolio_negative_return(self):
        curve   = np.linspace(10_000, 8_000, 100)
        metrics = portfolio_metrics(curve)
        assert metrics["total_return_pct"] < 0
        assert metrics["max_drawdown_pct"] < 0
