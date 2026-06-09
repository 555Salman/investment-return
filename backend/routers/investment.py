"""
Investment calculator endpoint.
Wraps invest.py logic — takes budget, years, risk and returns full projection.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
logging.disable(logging.CRITICAL)

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ml.models.lstm_model import LSTMForecaster
from ml.models.model_utils import load_model
from ml.optimization.portfolio_optimizer import optimize_portfolio, INTEREST_RATES

logging.disable(logging.NOTSET)

router = APIRouter(prefix="/api/investment", tags=["Investment"])

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PAIRS         = ["EUR_USD", "AUD_USD", "NZD_USD"]
PAIR_LABELS   = {
    "EUR_USD": "Euro (EUR)",
    "AUD_USD": "Australian Dollar (AUD)",
    "NZD_USD": "New Zealand Dollar (NZD)",
}

_model_cache:  dict[str, LSTMForecaster] = {}
_cache_mtimes: dict[str, float] = {}

_CHECKPOINTS_DIR = Path(__file__).resolve().parents[2] / "data" / "checkpoints"


def _load_models() -> dict[str, LSTMForecaster]:
    """Load models, reloading any whose checkpoint file has changed since last load."""
    global _model_cache, _cache_mtimes
    for pair in PAIRS:
        ckpt = _CHECKPOINTS_DIR / f"{pair}_best.pt"
        try:
            mtime = ckpt.stat().st_mtime
        except FileNotFoundError:
            continue
        if _model_cache.get(pair) is not None and _cache_mtimes.get(pair) == mtime:
            continue  # still fresh
        try:
            X = np.load(PROCESSED_DIR / pair / "X_test.npy")
            m = LSTMForecaster(input_size=X.shape[2])
            m = load_model(m, pair, tag="best")
            m.eval()
            _model_cache[pair]  = m
            _cache_mtimes[pair] = mtime
        except FileNotFoundError:
            pass
    return _model_cache


def _get_forecast_returns(models: dict) -> dict[str, float]:
    returns = {}
    for pair, model in models.items():
        X_test = np.load(PROCESSED_DIR / pair / "X_test.npy")
        y_test = np.load(PROCESSED_DIR / pair / "y_test.npy")
        last_window = torch.tensor(X_test[-1:], dtype=torch.float32)
        with torch.no_grad():
            predicted = float(model(last_window).item())
        last_actual = float(y_test[-1])
        returns[pair] = (predicted - last_actual) / (last_actual + 1e-8)
    return returns


# ── Schemas ────────────────────────────────────────────────────────────────────

class InvestmentRequest(BaseModel):
    budget:         float = Field(gt=0,  description="Total capital in USD")
    years:          float = Field(gt=0,  description="Investment period in years")
    risk_tolerance: str   = Field(default="medium", pattern="^(low|medium|high)$")
    pair_filter:    str   = Field(default="ALL",    description="EUR_USD | AUD_USD | NZD_USD | ALL")

class PairProjection(BaseModel):
    pair:             str
    label:            str
    deposit_usd:      float
    weight_pct:       float
    annual_rate_pct:  float
    interest_usd:     float
    fx_return_pct:    float
    fx_gain_loss_usd: float
    total_gain_usd:   float
    projected_usd:    float
    total_return_pct: float
    direction:        str   # "up" | "down"

class InvestmentResponse(BaseModel):
    budget:             float
    years:              float
    risk_tolerance:     str
    pair_filter:        str
    projections:        list[PairProjection]
    total_initial:      float
    total_interest:     float
    total_fx_gain:      float
    total_gain:         float
    total_projected:    float
    overall_return_pct: float
    forecast_returns:   dict[str, float]


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post("/calculate", response_model=InvestmentResponse)
def calculate(body: InvestmentRequest):
    """
    Calculate projected returns for a multicurrency deposit investment.
    """
    models = _load_models()
    if not models:
        raise HTTPException(
            status_code=503,
            detail="No trained models found. Run: python ml/training/train.py --pair_name EUR_USD"
        )

    investment_days  = int(body.years * 365)
    forecast_returns = _get_forecast_returns(models)

    # Apply pair filter
    active_pairs = (
        [body.pair_filter] if body.pair_filter != "ALL" and body.pair_filter in PAIRS
        else list(models.keys())
    )
    filtered_returns = {p: forecast_returns[p] for p in active_pairs if p in forecast_returns}

    try:
        result = optimize_portfolio(
            forecast_returns=filtered_returns,
            budget=body.budget,
            risk_tolerance=body.risk_tolerance,
            min_weight=0.05,
            max_weight=0.70,
            investment_days=investment_days,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Portfolio optimisation failed: {exc}")

    projections = []
    for pair in result["pairs"]:
        amount       = result["amounts"][pair]
        annual_rate  = INTEREST_RATES.get(pair, 0.02)
        interest     = amount * annual_rate * body.years
        fx_gain      = amount * filtered_returns[pair]
        total_gain   = interest + fx_gain
        fx_ret       = filtered_returns[pair]

        projections.append(PairProjection(
            pair             = pair,
            label            = PAIR_LABELS.get(pair, pair),
            deposit_usd      = round(amount, 2),
            weight_pct       = round(result["allocations"][pair] * 100, 2),
            annual_rate_pct  = round(annual_rate * 100, 2),
            interest_usd     = round(interest, 2),
            fx_return_pct    = round(fx_ret * 100, 4),
            fx_gain_loss_usd = round(fx_gain, 2),
            total_gain_usd   = round(total_gain, 2),
            projected_usd    = round(amount + total_gain, 2),
            total_return_pct = round((total_gain / amount) * 100 if amount else 0, 4),
            direction        = "up" if fx_ret >= 0 else "down",
        ))

    total_interest  = sum(p.interest_usd     for p in projections)
    total_fx        = sum(p.fx_gain_loss_usd for p in projections)
    total_gain_all  = sum(p.total_gain_usd   for p in projections)
    total_projected = body.budget + total_gain_all

    return InvestmentResponse(
        budget             = body.budget,
        years              = body.years,
        risk_tolerance     = body.risk_tolerance,
        pair_filter        = body.pair_filter,
        projections        = projections,
        total_initial      = body.budget,
        total_interest     = round(total_interest, 2),
        total_fx_gain      = round(total_fx, 2),
        total_gain         = round(total_gain_all, 2),
        total_projected    = round(total_projected, 2),
        overall_return_pct = round((total_gain_all / body.budget) * 100 if body.budget else 0, 4),
        forecast_returns   = {p: round(v, 6) for p, v in filtered_returns.items()},
    )
