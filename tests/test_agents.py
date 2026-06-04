"""
Unit tests for the agentic AI layer.
Run with: python -m pytest tests/test_agents.py -v
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from ml.agents.base_agent import BaseAgent, AgentStatus
from ml.agents.decision_agent import DecisionAgent
from ml.agents.rebalance_agent import RebalanceAgent


# ── BaseAgent ──────────────────────────────────────────────────────────────────

class TestBaseAgent:

    def test_initial_status_is_idle(self):
        agent = BaseAgent("test")
        assert agent.status == AgentStatus.IDLE

    def test_record_appends_to_log(self):
        agent = BaseAgent("test")
        agent._record("action_a", {"x": 1})
        agent._record("action_b")
        assert len(agent.log) == 2
        assert agent.log[0]["action"] == "action_a"
        assert agent.log[1]["action"] == "action_b"

    def test_log_entry_has_required_keys(self):
        agent = BaseAgent("test")
        entry = agent._record("something", {"val": 42})
        assert {"timestamp", "agent", "action", "detail"} <= set(entry.keys())


# ── DecisionAgent ──────────────────────────────────────────────────────────────

class TestDecisionAgent:

    MOCK_FORECASTS = {
        "EUR_USD": {"forecast_return": 0.03, "predicted_next": 1.12, "last_actual": 1.09},
        "AUD_USD": {"forecast_return": 0.04, "predicted_next": 0.67, "last_actual": 0.64},
        "NZD_USD": {"forecast_return": 0.025, "predicted_next": 0.62, "last_actual": 0.60},
    }

    def test_decision_returns_allocations(self, monkeypatch):
        monkeypatch.setattr(
            "ml.optimization.portfolio_optimizer.get_risk_limits",
            lambda pairs, risk_tolerance: {p: 0.01 for p in pairs},
        )
        agent  = DecisionAgent(budget=10_000)
        result = asyncio.run(agent.run(self.MOCK_FORECASTS))
        assert "allocations" in result
        assert set(result["allocations"].keys()) == set(self.MOCK_FORECASTS.keys())

    def test_first_run_always_triggers_rebalance(self, monkeypatch):
        monkeypatch.setattr(
            "ml.optimization.portfolio_optimizer.get_risk_limits",
            lambda pairs, risk_tolerance: {p: 0.01 for p in pairs},
        )
        agent  = DecisionAgent()
        result = asyncio.run(agent.run(self.MOCK_FORECASTS))
        assert result["rebalance_needed"] is True

    def test_no_rebalance_when_unchanged(self, monkeypatch):
        monkeypatch.setattr(
            "ml.optimization.portfolio_optimizer.get_risk_limits",
            lambda pairs, risk_tolerance: {p: 0.01 for p in pairs},
        )
        agent  = DecisionAgent()
        result = asyncio.run(agent.run(self.MOCK_FORECASTS))
        # Set current portfolio to match proposed
        agent.current_portfolio = result["allocations"]
        # Run again with same forecasts → no drift
        result2 = asyncio.run(agent.run(self.MOCK_FORECASTS))
        assert result2["rebalance_needed"] is False


# ── RebalanceAgent ─────────────────────────────────────────────────────────────

class TestRebalanceAgent:

    MOCK_DECISION = {
        "rebalance_needed":    True,
        "allocations":         {"EUR_USD": 0.4, "AUD_USD": 0.35, "NZD_USD": 0.25},
        "amounts":             {"EUR_USD": 4000, "AUD_USD": 3500, "NZD_USD": 2500},
        "budget":              10_000,
        "expected_return_pct": 0.05,
        "drifts":              {},
    }

    def test_rebalance_updates_history(self):
        agent  = RebalanceAgent(simulation=True)
        result = asyncio.run(agent.run(self.MOCK_DECISION))
        assert result["rebalanced"] is True
        assert len(agent.portfolio_history) == 1

    def test_no_rebalance_when_not_needed(self):
        agent    = RebalanceAgent(simulation=True)
        decision = {**self.MOCK_DECISION, "rebalance_needed": False}
        result   = asyncio.run(agent.run(decision))
        assert result["rebalanced"] is False

    def test_trades_computed_for_all_pairs(self):
        agent  = RebalanceAgent(simulation=True)
        result = asyncio.run(agent.run(self.MOCK_DECISION))
        assert set(result["trades"].keys()) == set(self.MOCK_DECISION["allocations"].keys())

    def test_trade_actions_valid(self):
        agent  = RebalanceAgent(simulation=True)
        result = asyncio.run(agent.run(self.MOCK_DECISION))
        valid  = {"BUY", "SELL", "HOLD"}
        for trade in result["trades"].values():
            assert trade["action"] in valid
