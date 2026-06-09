"""
Portfolio optimization endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException

from backend.core.security import get_current_user
from backend.models.schemas import OptimizeRequest
from backend.services.forecast_service import forecast_service
from backend.services.portfolio_service import portfolio_service

router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])


async def _get_forecast_returns() -> dict[str, float]:
    """Helper — fetch LSTM forecast returns, raise 503 if models missing."""
    data = await forecast_service.get_all_forecasts()
    if not data["forecasts"]:
        raise HTTPException(status_code=503, detail="No trained models. Train models first.")
    return {pair: v["forecast_return"] for pair, v in data["forecasts"].items()}


@router.post("/optimize", summary="Run LP portfolio optimization")
async def optimize_portfolio(body: OptimizeRequest, _: dict = Depends(get_current_user)):
    """
    Runs the linear programming optimizer using LSTM forecast returns.
    Returns optimal capital allocation across the three currency pairs.
    """
    forecast_returns = await _get_forecast_returns()
    return await portfolio_service.optimize(
        forecast_returns=forecast_returns,
        budget=body.budget,
        risk_tolerance=body.risk_tolerance,
        min_weight=body.min_weight,
        max_weight=body.max_weight,
        investment_days=body.investment_days,
    )


@router.get("/benchmark", summary="Compare LP vs equal-weight strategy")
async def benchmark(
    budget: float = 10_000.0,
    investment_days: int = 365,
    risk_tolerance: str = "medium",
    _: dict = Depends(get_current_user),
):
    """
    Returns a side-by-side comparison of the LP-optimised strategy
    vs a naive equal-weight benchmark.
    """
    forecast_returns = await _get_forecast_returns()
    return await portfolio_service.benchmark(
        forecast_returns=forecast_returns,
        budget=budget,
        investment_days=investment_days,
        risk_tolerance=risk_tolerance,
    )
