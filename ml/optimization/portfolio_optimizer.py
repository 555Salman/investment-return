"""
Portfolio optimizer using Linear Programming.

Objective:
    Maximize  Σ (forecast_return_i + interest_rate_i) × allocation_i

Constraints:
    1. Σ allocation_i = 1                          (fully invested)
    2. allocation_i >= min_weight                  (no shorting)
    3. allocation_i <= max_weight                  (diversification cap)
    4. volatility_i × allocation_i <= risk_limit_i (per-pair risk cap)

Uses scipy.optimize.linprog (minimises, so we negate the objective).
"""

import logging
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.optimization.risk_manager import get_risk_limits

logger = logging.getLogger(__name__)

# ── Sri Lankan bank deposit interest rates (annual %) for foreign currency accounts
# Source: indicative rates; update from Bank of Ceylon / Peoples Bank as needed
INTEREST_RATES = {
    "EUR_USD": 0.0210,   # 2.10% p.a.
    "AUD_USD": 0.0380,   # 3.80% p.a.
    "NZD_USD": 0.0400,   # 4.00% p.a.
}


def optimize_portfolio(
    forecast_returns: dict[str, float],
    budget:           float  = 10_000.0,
    risk_tolerance:   str    = "medium",
    min_weight:       float  = 0.05,
    max_weight:       float  = 0.60,
    investment_days:  int    = 365,
) -> dict:
    """
    Solve the LP to find optimal capital allocation across currency pairs.

    Args:
        forecast_returns: dict mapping pair_name → expected return over investment_days
                          (output from LSTM forecast, expressed as a fraction e.g. 0.03)
        budget:           Total capital to allocate (in USD)
        risk_tolerance:   "low" | "medium" | "high"
        min_weight:       Minimum allocation per pair (e.g. 0.05 = 5%)
        max_weight:       Maximum allocation per pair (e.g. 0.60 = 60%)
        investment_days:  Holding period in days (used to scale interest rates)

    Returns:
        dict with keys: allocations, amounts, expected_return, expected_amount,
                        total_return_pct, risk_limits, status
    """
    pairs = list(forecast_returns.keys())
    n     = len(pairs)

    # ── Interest rates scaled to investment period ──
    daily_rates = {p: INTEREST_RATES.get(p, 0.02) / 365 for p in pairs}
    period_interest = {p: daily_rates[p] * investment_days for p in pairs}

    # ── Total expected return per pair ──
    total_returns = np.array([
        forecast_returns[p] + period_interest[p]
        for p in pairs
    ])

    logger.info("Expected returns per pair:")
    for p, r in zip(pairs, total_returns):
        logger.info(f"  {p}: forecast={forecast_returns[p]:.4f}  "
                    f"interest={period_interest[p]:.4f}  total={r:.4f}")

    # ── Risk limits ──
    risk_limits = get_risk_limits(pairs, risk_tolerance)
    volatilities = np.array([
        risk_limits[p] / (1.5 if risk_tolerance == "medium" else
                          1.0 if risk_tolerance == "low" else 2.5)
        for p in pairs
    ])

    # ── LP setup ──
    # linprog minimises, so negate returns for maximisation
    c = -total_returns

    # Inequality constraints: A_ub @ x <= b_ub
    # vol_i * x_i <= risk_limit_i  →  diag(vol) @ x <= risk_limits
    A_ub = np.diag(volatilities)
    b_ub = np.array([risk_limits[p] for p in pairs])

    # Equality constraint: Σ x_i = 1
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])

    # Bounds: min_weight <= x_i <= max_weight
    bounds = [(min_weight, max_weight)] * n

    result = linprog(
        c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if result.status != 0 or result.x is None:
        raise ValueError(
            f"LP infeasible (status={result.status}): {result.message}"
        )

    weights = result.x
    weights = np.clip(weights, min_weight, max_weight)
    weights /= weights.sum()   # re-normalise after clipping

    expected_return_pct = float(np.dot(weights, total_returns))
    expected_amount     = budget * (1 + expected_return_pct)

    allocations = {p: float(w) for p, w in zip(pairs, weights)}
    amounts     = {p: float(w * budget) for p, w in zip(pairs, weights)}

    logger.info(f"\nOptimal allocation (budget=${budget:,.2f}):")
    for p in pairs:
        logger.info(f"  {p}: {allocations[p]*100:.1f}%  =  ${amounts[p]:,.2f}")
    logger.info(f"  Expected return: {expected_return_pct*100:.2f}%  →  ${expected_amount:,.2f}")

    return {
        "pairs":              pairs,
        "allocations":        allocations,
        "amounts":            amounts,
        "expected_return_pct": expected_return_pct,
        "expected_amount":    expected_amount,
        "budget":             budget,
        "risk_limits":        risk_limits,
        "solver_status":      result.message,
        "investment_days":    investment_days,
    }


# ── Benchmark strategies ───────────────────────────────────────────────────────

def equal_weight_portfolio(
    forecast_returns: dict[str, float],
    budget: float = 10_000.0,
    investment_days: int = 365,
) -> dict:
    """Equal allocation across all pairs (naive benchmark)."""
    pairs   = list(forecast_returns.keys())
    n       = len(pairs)
    weight  = 1.0 / n

    daily_rates     = {p: INTEREST_RATES.get(p, 0.02) / 365 for p in pairs}
    period_interest = {p: daily_rates[p] * investment_days for p in pairs}
    total_returns   = {p: forecast_returns[p] + period_interest[p] for p in pairs}

    expected_return_pct = sum(weight * total_returns[p] for p in pairs)

    return {
        "strategy":           "equal_weight",
        "pairs":              pairs,
        "allocations":        {p: weight for p in pairs},
        "amounts":            {p: weight * budget for p in pairs},
        "expected_return_pct": expected_return_pct,
        "expected_amount":    budget * (1 + expected_return_pct),
        "budget":             budget,
    }


def compare_strategies(
    forecast_returns: dict[str, float],
    budget: float = 10_000.0,
    investment_days: int = 365,
    risk_tolerance: str = "medium",
) -> list[dict]:
    """Run both strategies and return a comparison list."""
    lp_result  = optimize_portfolio(forecast_returns, budget, risk_tolerance,
                                    investment_days=investment_days)
    eq_result  = equal_weight_portfolio(forecast_returns, budget, investment_days)

    lp_result["strategy"]  = "lp_optimised"
    eq_result["strategy"]  = "equal_weight"

    return [lp_result, eq_result]
