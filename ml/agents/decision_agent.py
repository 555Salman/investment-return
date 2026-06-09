"""
Decision Agent.
Consumes LSTM forecasts, runs the LP optimizer, and recommends
a portfolio allocation. Flags if the recommended allocation differs
significantly from the current one (rebalance trigger).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.agents.base_agent import BaseAgent, AgentStatus
from ml.optimization.portfolio_optimizer import optimize_portfolio

REBALANCE_THRESHOLD = 0.05   # trigger rebalance if any weight drifts by > 5%


class DecisionAgent(BaseAgent):
    """
    Runs the LP portfolio optimizer on the latest forecasts
    and determines whether a portfolio rebalance is needed.
    """

    def __init__(
        self,
        budget:          float = 10_000.0,
        risk_tolerance:  str   = "medium",
        min_weight:      float = 0.05,
        max_weight:      float = 0.60,
        investment_days: int   = 365,
    ):
        super().__init__("DecisionAgent")
        self.budget          = budget
        self.risk_tolerance  = risk_tolerance
        self.min_weight      = min_weight
        self.max_weight      = max_weight
        self.investment_days = investment_days
        self.current_portfolio: dict[str, float] = {}

    def _needs_rebalance(self, proposed: dict[str, float]) -> tuple[bool, dict]:
        """
        Compare proposed allocations to current holdings.
        Returns (should_rebalance, drifts_dict).
        """
        if not self.current_portfolio:
            return True, {}   # no existing portfolio → always allocate

        drifts = {
            pair: abs(proposed[pair] - self.current_portfolio.get(pair, 0.0))
            for pair in proposed
        }
        if not drifts:
            return True, {}   # no pairs in proposal → trigger rebalance as safety default

        max_drift = max(drifts.values())
        return max_drift >= REBALANCE_THRESHOLD, drifts

    async def run(self, forecasts: dict[str, dict]) -> dict:
        """
        Args:
            forecasts: output from ForecastAgent.run()
                       {pair: {forecast_return, predicted_next, last_actual}}

        Returns:
            decision dict with keys: allocations, amounts, rebalance_needed, drifts,
                                     expected_return_pct, solver_status
        """
        self.status = AgentStatus.RUNNING
        self._record("decision_started", {"pairs": list(forecasts.keys())})

        forecast_returns = {
            pair: data["forecast_return"]
            for pair, data in forecasts.items()
        }

        result = optimize_portfolio(
            forecast_returns=forecast_returns,
            budget=self.budget,
            risk_tolerance=self.risk_tolerance,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            investment_days=self.investment_days,
        )

        proposed        = result["allocations"]
        rebalance, drifts = self._needs_rebalance(proposed)

        decision = {
            **result,
            "rebalance_needed": rebalance,
            "drifts":           drifts,
        }

        self._record("decision_ready", {
            "rebalance_needed":    rebalance,
            "expected_return_pct": f"{result['expected_return_pct']*100:.2f}%",
            "allocations":         {p: f"{w*100:.1f}%" for p, w in proposed.items()},
        })

        # Track latest recommended allocation so future drift checks have a valid baseline
        self.current_portfolio = proposed

        self.status = AgentStatus.IDLE
        return decision
