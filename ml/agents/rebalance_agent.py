"""
Rebalance Agent.
Receives a decision from the DecisionAgent and either executes the
rebalance (simulation mode) or logs a recommendation for the investor.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.agents.base_agent import BaseAgent, AgentStatus


class RebalanceAgent(BaseAgent):
    """
    Simulates portfolio rebalancing based on the DecisionAgent's output.

    In simulation mode it:
      - Computes trades needed (buy / sell per pair)
      - Updates the portfolio state
      - Logs the rebalance event

    In production this would connect to a bank/brokerage API.
    """

    def __init__(self, simulation: bool = True):
        super().__init__("RebalanceAgent")
        self.simulation = simulation
        self.portfolio_history: list[dict] = []

    def _compute_trades(
        self,
        current: dict[str, float],
        proposed: dict[str, float],
        budget: float,
    ) -> dict[str, dict]:
        """
        Compute the USD amount to buy (+) or sell (-) for each pair.
        """
        trades = {}
        for pair in proposed:
            current_amount  = current.get(pair, 0.0) * budget
            proposed_amount = proposed[pair] * budget
            delta           = proposed_amount - current_amount
            trades[pair] = {
                "current_pct":  current.get(pair, 0.0) * 100,
                "proposed_pct": proposed[pair] * 100,
                "delta_usd":    round(delta, 2),
                "action":       "BUY" if delta > 0 else ("SELL" if delta < 0 else "HOLD"),
            }
        return trades

    async def run(self, decision: dict) -> dict:
        """
        Args:
            decision: output from DecisionAgent.run()

        Returns:
            rebalance_result dict with trades and updated portfolio state
        """
        self.status = AgentStatus.RUNNING

        if not decision.get("rebalance_needed", False):
            self._record("no_rebalance_needed")
            self.status = AgentStatus.IDLE
            return {"rebalanced": False, "reason": "No significant drift detected."}

        proposed = decision["allocations"]
        budget   = decision["budget"]

        # Use last known portfolio state (or empty if first run)
        current  = self.portfolio_history[-1]["allocations"] if self.portfolio_history else {}
        trades   = self._compute_trades(current, proposed, budget)

        snapshot = {
            "timestamp":   datetime.utcnow().isoformat(),
            "allocations": proposed,
            "amounts":     decision["amounts"],
            "trades":      trades,
            "budget":      budget,
            "expected_return_pct": decision["expected_return_pct"],
        }
        self.portfolio_history.append(snapshot)

        if self.simulation:
            self._record("rebalance_simulated", {
                "trades": {p: t["action"] + f" ${abs(t['delta_usd']):,.2f}" for p, t in trades.items()}
            })
        else:
            # Production: call bank API here
            self._record("rebalance_executed", {"trades": trades})

        self.status = AgentStatus.IDLE
        return {"rebalanced": True, "trades": trades, "snapshot": snapshot}
