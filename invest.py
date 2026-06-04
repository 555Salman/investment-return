"""
Multicurrency Investment Calculator
====================================
Enter your total capital and investment period.
The system will forecast exchange rate movements, optimise the allocation
across EUR, AUD, and NZD deposits, and show the projected return.

Usage:
    python invest.py --budget 10000000 --years 2 --risk medium
    python invest.py --budget 500000   --years 1 --risk low
    python invest.py                              (interactive prompts)
"""

import argparse
import sys
import logging
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.disable(logging.CRITICAL)   # suppress internal logs for clean output

import numpy as np
import torch

from ml.models.lstm_model        import LSTMForecaster
from ml.models.model_utils       import load_model
from ml.optimization.portfolio_optimizer import optimize_portfolio, INTEREST_RATES

PROCESSED_DIR = Path("data/processed")
PAIRS         = ["EUR_USD", "AUD_USD", "NZD_USD"]
PAIR_LABELS   = {"EUR_USD": "Euro (EUR)", "AUD_USD": "Australian Dollar (AUD)", "NZD_USD": "New Zealand Dollar (NZD)"}


# ── 1. Load LSTM models ────────────────────────────────────────────────────────

def load_models() -> dict:
    models = {}
    for pair in PAIRS:
        try:
            X = np.load(PROCESSED_DIR / pair / "X_test.npy")
            m = LSTMForecaster(input_size=X.shape[2])
            m = load_model(m, pair, tag="best")
            m.eval()
            models[pair] = m
        except FileNotFoundError:
            pass
    return models


# ── 2. Forecast returns ────────────────────────────────────────────────────────

def get_forecast_returns(models: dict) -> dict[str, float]:
    """
    Use each LSTM model to predict the next exchange rate
    and compute expected capital gain/loss as a fraction.
    """
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


# ── 3. Compute detailed projection per pair ────────────────────────────────────

def compute_projection(
    pair:            str,
    amount_usd:      float,
    years:           float,
    forecast_return: float,
) -> dict:
    """
    For a given USD deposit amount, compute:
      - Interest earned (from bank deposit rate)
      - Exchange rate gain/loss (from LSTM forecast)
      - Total projected value
    """
    annual_rate   = INTEREST_RATES.get(pair, 0.02)
    interest_frac = annual_rate * years
    fx_frac       = forecast_return           # LSTM 1-step extrapolated

    interest_earned  = amount_usd * interest_frac
    fx_gain_loss     = amount_usd * fx_frac
    total_gain       = interest_earned + fx_gain_loss
    projected_value  = amount_usd + total_gain

    return {
        "pair":             pair,
        "label":            PAIR_LABELS[pair],
        "deposit_usd":      amount_usd,
        "annual_rate_pct":  annual_rate * 100,
        "interest_usd":     interest_earned,
        "fx_return_pct":    fx_frac * 100,
        "fx_gain_loss_usd": fx_gain_loss,
        "total_gain_usd":   total_gain,
        "projected_usd":    projected_value,
        "total_return_pct": (total_gain / amount_usd) * 100 if amount_usd else 0,
    }


# ── 4. Print report ────────────────────────────────────────────────────────────

def print_report(
    budget:       float,
    years:        float,
    risk:         str,
    allocations:  dict,
    amounts:      dict,
    projections:  list[dict],
    total_proj:   float,
    forecast_returns: dict,
) -> None:

    SEP  = "=" * 72
    SEP2 = "-" * 72

    print(f"\n{SEP}")
    print("  MULTICURRENCY INVESTMENT PROJECTION")
    print(SEP)
    print(f"  Total Capital      : ${budget:>20,.2f}")
    print(f"  Investment Period  : {years} year{'s' if years != 1 else ''}")
    print(f"  Risk Tolerance     : {risk.title()}")
    print(f"  Strategy           : LP-Optimised (LSTM + Linear Programming)")
    print(SEP)

    print(f"\n{'CURRENCY':<28} {'ALLOCATION':>10} {'DEPOSIT (USD)':>16} {'ANNUAL RATE':>12}")
    print(SEP2)
    for p in projections:
        print(
            f"  {p['label']:<26} "
            f"{allocations[p['pair']]*100:>9.1f}%  "
            f"${p['deposit_usd']:>14,.2f}  "
            f"{p['annual_rate_pct']:>10.2f}%"
        )
    print(SEP2)

    print(f"\n{'CURRENCY':<28} {'INTEREST':>14} {'FX GAIN/LOSS':>14} {'TOTAL GAIN':>14}")
    print(SEP2)
    for p in projections:
        fx_sign    = "+" if p['fx_gain_loss_usd'] >= 0 else ""
        int_sign   = "+"
        total_sign = "+" if p['total_gain_usd'] >= 0 else ""
        print(
            f"  {p['label']:<26} "
            f"{int_sign}${p['interest_usd']:>12,.2f}  "
            f"{fx_sign}${p['fx_gain_loss_usd']:>12,.2f}  "
            f"{total_sign}${p['total_gain_usd']:>12,.2f}"
        )
    print(SEP2)

    total_gain  = total_proj - budget
    return_pct  = (total_gain / budget) * 100
    gain_sign   = "+" if total_gain >= 0 else ""

    print(f"\n  Initial Capital    : ${budget:>20,.2f}")
    print(f"  Total Gain / Loss  : {gain_sign}${total_gain:>19,.2f}  ({gain_sign}{return_pct:.2f}%)")
    print(f"  Projected Value    : ${total_proj:>20,.2f}")
    print(SEP)

    print("\n  EXCHANGE RATE FORECAST (LSTM)")
    print(SEP2)
    print(f"  {'Pair':<12} {'Direction':>12} {'Forecast Move':>15}")
    print(SEP2)
    for pair, ret in forecast_returns.items():
        direction = "APPRECIATE (+)" if ret >= 0 else "DEPRECIATE (-)"
        color     = "+" if ret >= 0 else ""
        print(f"  {pair.replace('_','/'):<12} {direction:>12}   {color}{ret*100:.3f}%")
    print(SEP2)
    print()


# ── 5. Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multicurrency Investment Calculator")
    parser.add_argument("--budget", type=float, default=None, help="Total capital in USD")
    parser.add_argument("--years",  type=float, default=None, help="Investment period in years")
    parser.add_argument("--risk",   type=str,   default=None, choices=["low", "medium", "high"])
    args = parser.parse_args()

    # Interactive prompts if not provided via CLI
    if args.budget is None:
        args.budget = float(input("  Enter total capital (USD): $").replace(",", ""))
    if args.years is None:
        args.years  = float(input("  Enter investment period (years): "))
    if args.risk is None:
        raw = input("  Enter risk tolerance [low / medium / high] (default: medium): ").strip().lower()
        args.risk = raw if raw in ("low", "medium", "high") else "medium"

    investment_days = int(args.years * 365)

    print("\n  Loading models and computing forecasts…")

    # Load LSTM models
    models = load_models()
    if not models:
        print("\n  ERROR: No trained models found.")
        print("  Run these first:")
        for pair in PAIRS:
            print(f"    python ml/training/train.py --pair_name {pair} --epochs 50")
        sys.exit(1)

    missing = [p for p in PAIRS if p not in models]
    if missing:
        print(f"\n  WARNING: Models missing for {missing}. Results will only cover trained pairs.")

    # Forecast
    forecast_returns = get_forecast_returns(models)

    # LP Optimization
    result = optimize_portfolio(
        forecast_returns=forecast_returns,
        budget=args.budget,
        risk_tolerance=args.risk,
        min_weight=0.05,
        max_weight=0.70,
        investment_days=investment_days,
    )

    # Per-pair projections
    projections = []
    total_projected = 0.0
    for pair in result["pairs"]:
        proj = compute_projection(
            pair=pair,
            amount_usd=result["amounts"][pair],
            years=args.years,
            forecast_return=forecast_returns[pair],
        )
        projections.append(proj)
        total_projected += proj["projected_usd"]

    print_report(
        budget=args.budget,
        years=args.years,
        risk=args.risk,
        allocations=result["allocations"],
        amounts=result["amounts"],
        projections=projections,
        total_proj=total_projected,
        forecast_returns=forecast_returns,
    )


if __name__ == "__main__":
    main()
