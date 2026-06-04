"""
Portfolio service — wraps the LP optimizer for use inside FastAPI.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.optimization.portfolio_optimizer import optimize_portfolio, compare_strategies


class PortfolioService:

    async def optimize(
        self,
        forecast_returns: dict[str, float],
        budget:           float,
        risk_tolerance:   str,
        min_weight:       float,
        max_weight:       float,
        investment_days:  int,
    ) -> dict:
        result = optimize_portfolio(
            forecast_returns=forecast_returns,
            budget=budget,
            risk_tolerance=risk_tolerance,
            min_weight=min_weight,
            max_weight=max_weight,
            investment_days=investment_days,
        )

        # Shape allocations for the API response
        allocations = {
            pair: {
                "weight_pct": round(w * 100, 2),
                "amount_usd": round(result["amounts"][pair], 2),
            }
            for pair, w in result["allocations"].items()
        }

        return {
            "allocations":         allocations,
            "expected_return_pct": round(result["expected_return_pct"] * 100, 4),
            "expected_amount":     round(result["expected_amount"], 2),
            "budget":              budget,
            "risk_tolerance":      risk_tolerance,
            "solver_status":       result["solver_status"],
            "generated_at":        datetime.now(timezone.utc).isoformat(),
        }

    async def benchmark(
        self,
        forecast_returns: dict[str, float],
        budget: float,
        investment_days: int,
        risk_tolerance: str,
    ) -> list[dict]:
        strategies = compare_strategies(
            forecast_returns=forecast_returns,
            budget=budget,
            investment_days=investment_days,
            risk_tolerance=risk_tolerance,
        )
        return [
            {
                "strategy":            s["strategy"],
                "expected_return_pct": round(s["expected_return_pct"] * 100, 4),
                "expected_amount":     round(s["expected_amount"], 2),
                "allocations":         {p: round(w * 100, 2) for p, w in s["allocations"].items()},
            }
            for s in strategies
        ]


portfolio_service = PortfolioService()
