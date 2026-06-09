"""
Walk-forward backtesting engine.

Simulates the full pipeline on the test window:
  For each day t in the test set:
    1. Use LSTM to forecast next-day price
    2. Run LP optimizer with that forecast
    3. Allocate capital accordingly
    4. Observe actual next-day return
    5. Update portfolio value

Three strategies are compared:
  - lp_optimised   : LP allocation driven by LSTM forecasts
  - equal_weight   : Fixed 1/3 across all three pairs
  - buy_and_hold   : Allocate once on day 0, never rebalance

Usage:
    python ml/training/backtest.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.models.lstm_model    import LSTMForecaster
from ml.models.model_utils   import load_model
from ml.optimization.portfolio_optimizer import optimize_portfolio, INTEREST_RATES
from ml.training.evaluate    import portfolio_metrics, metrics_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PAIRS         = ["EUR_USD", "AUD_USD", "NZD_USD"]
RESULTS_DIR   = Path(__file__).resolve().parents[2] / "data" / "backtest_results"


# ── Load models and data ───────────────────────────────────────────────────────

def _load_models() -> dict[str, LSTMForecaster]:
    models = {}
    for pair in PAIRS:
        try:
            X = np.load(PROCESSED_DIR / pair / "X_test.npy")
            m = LSTMForecaster(input_size=X.shape[2])
            m = load_model(m, pair, tag="best")
            m.eval()
            models[pair] = m
            logger.info(f"  Loaded model: {pair}")
        except FileNotFoundError:
            logger.warning(f"  No model for {pair} — run training first")
    return models


def _load_test_data() -> dict[str, dict]:
    """Load X_test sequences and y_test actual prices for all pairs."""
    data = {}
    for pair in PAIRS:
        base = PROCESSED_DIR / pair
        data[pair] = {
            "X": np.load(base / "X_test.npy"),
            "y": np.load(base / "y_test.npy"),
        }
    return data


# ── Per-step forecast ──────────────────────────────────────────────────────────

def _forecast_step(
    models: dict[str, LSTMForecaster],
    test_data: dict,
    t: int,
) -> dict[str, float]:
    """
    Run LSTM inference at step t for all pairs.
    Returns forecast_return per pair: (predicted_next - current_price) / current_price.

    y[t] is the TARGET for window X[t] (the price one step ahead of the window).
    The "current" known price is y[t-1]; for t=0 we fall back to y[0].
    """
    returns = {}
    for pair, model in models.items():
        X = torch.tensor(test_data[pair]["X"][t : t + 1], dtype=torch.float32)
        with torch.no_grad():
            pred = float(model(X).cpu().item())
        current = float(test_data[pair]["y"][t - 1] if t > 0 else test_data[pair]["y"][t])
        returns[pair] = (pred - current) / (current + 1e-8)
    return returns


# ── Backtesting engine ─────────────────────────────────────────────────────────

def run_backtest(
    budget:          float = 10_000.0,
    risk_tolerance:  str   = "medium",
    rebalance_every: int   = 1,         # days between rebalances (1 = daily)
    risk_free_rate:  float = 0.02,
) -> dict:
    """
    Walk-forward backtest over the test set.

    Args:
        budget:           Starting capital (USD)
        risk_tolerance:   "low" | "medium" | "high"
        rebalance_every:  How often to rebalance (1 = every step, 5 = weekly, 21 = monthly)
        risk_free_rate:   Annual risk-free rate for Sharpe calculation

    Returns:
        dict with equity curves and performance metrics for all three strategies
    """
    logger.info("Loading models and test data…")
    models    = _load_models()
    test_data = _load_test_data()

    if not models:
        raise RuntimeError("No trained models found. Run training first.")

    n_steps = min(len(test_data[p]["y"]) for p in models) - 1
    logger.info(f"Running backtest over {n_steps} test steps")

    # ── Equity curves ──
    lp_equity  = np.zeros(n_steps + 1)
    ew_equity  = np.zeros(n_steps + 1)
    bnh_equity = np.zeros(n_steps + 1)

    lp_equity[0] = ew_equity[0] = bnh_equity[0] = budget

    # Buy-and-hold: fix allocation on day 0 equally
    bnh_weights = {p: 1.0 / len(models) for p in models}

    # LP: start equal-weight; updated AFTER each step so weights always lag
    # one step behind the return they are applied to (no look-ahead).
    lp_weights   = {p: 1.0 / len(models) for p in models}
    lp_fail_count = 0   # track consecutive LP failures for stale-weight detection

    # Daily interest rates
    daily_interest = {p: INTEREST_RATES.get(p, 0.02) / 365 for p in models}

    for t in range(n_steps):
        # ── Step 1: Earn actual returns using weights decided at the PREVIOUS step ──
        # This ordering eliminates look-ahead: we cannot use today's forecast to
        # earn today's return because the forecast is computed AFTER this block.
        actual_returns = {}
        for pair in models:
            y_t  = float(test_data[pair]["y"][t])
            y_t1 = float(test_data[pair]["y"][t + 1])
            actual_returns[pair] = (y_t1 - y_t) / (y_t + 1e-8) + daily_interest[pair]

        lp_return  = sum(lp_weights[p]  * actual_returns[p] for p in models)
        ew_return  = sum((1 / len(models)) * actual_returns[p] for p in models)
        bnh_return = sum(bnh_weights[p]  * actual_returns[p] for p in models)

        lp_equity[t + 1]  = lp_equity[t]  * (1 + lp_return)
        ew_equity[t + 1]  = ew_equity[t]  * (1 + ew_return)
        bnh_equity[t + 1] = bnh_equity[t] * (1 + bnh_return)

        # ── Step 2: Forecast and update weights for the NEXT step ──
        if t % rebalance_every == 0:
            try:
                forecast_returns = _forecast_step(models, test_data, t)
                result = optimize_portfolio(
                    forecast_returns=forecast_returns,
                    budget=lp_equity[t + 1],
                    risk_tolerance=risk_tolerance,
                )
                lp_weights    = result["allocations"]
                lp_fail_count = 0
            except Exception as exc:
                lp_fail_count += 1
                if lp_fail_count >= 3:
                    lp_weights    = {p: 1.0 / len(models) for p in models}
                    lp_fail_count = 0
                    logger.warning(
                        f"  LP failed 3× in a row at t={t}: {exc} — reset to equal weights"
                    )
                else:
                    logger.warning(
                        f"  LP failed at t={t} ({lp_fail_count}/3): {exc} — keeping last weights"
                    )

        if (t + 1) % 20 == 0:
            logger.info(
                f"  Step {t+1:>3}/{n_steps}  "
                f"LP=${lp_equity[t+1]:>9.2f}  "
                f"EW=${ew_equity[t+1]:>9.2f}  "
                f"BnH=${bnh_equity[t+1]:>9.2f}"
            )

    # ── LSTM forecast accuracy per pair ──
    forecast_metrics = {}
    for pair, model in models.items():
        X   = torch.tensor(test_data[pair]["X"], dtype=torch.float32)
        y   = test_data[pair]["y"]
        with torch.no_grad():
            preds = model(X).cpu().numpy().squeeze()
        forecast_metrics[pair] = metrics_report(pair, y, preds)

    # ── Portfolio performance ──
    results = {
        "n_steps": n_steps,
        "budget":  budget,
        "equity_curves": {
            "lp_optimised":  lp_equity.tolist(),
            "equal_weight":  ew_equity.tolist(),
            "buy_and_hold":  bnh_equity.tolist(),
        },
        "performance": {
            "lp_optimised":  portfolio_metrics(lp_equity,  risk_free_rate),
            "equal_weight":  portfolio_metrics(ew_equity,  risk_free_rate),
            "buy_and_hold":  portfolio_metrics(bnh_equity, risk_free_rate),
        },
        "forecast_metrics": forecast_metrics,
    }

    _save_results(results)
    _print_summary(results)
    return results


# ── Output ─────────────────────────────────────────────────────────────────────

def _print_summary(results: dict) -> None:
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Total Return':>14} {'Sharpe':>8} {'Max DD':>10} {'Final $':>12}")
    print("-" * 70)
    for strategy, m in results["performance"].items():
        print(
            f"{strategy:<20} "
            f"{m['total_return_pct']:>13.2f}%  "
            f"{m['sharpe_ratio']:>8.3f}  "
            f"{m['max_drawdown_pct']:>9.2f}%  "
            f"${m['final_value']:>11,.2f}"
        )
    print("=" * 70)

    print("\nFORECAST ACCURACY")
    print("-" * 50)
    print(f"{'Pair':<12} {'RMSE':>10} {'MAE':>10} {'Dir Acc':>10}")
    for pair, m in results["forecast_metrics"].items():
        print(
            f"{pair:<12} "
            f"{m['rmse']:>10.6f}  "
            f"{m['mae']:>10.6f}  "
            f"{m['directional_accuracy']*100:>9.1f}%"
        )
    print()


def _save_results(results: dict) -> None:
    import json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save equity curves as CSV
    curves = results["equity_curves"]
    df = pd.DataFrame(curves)
    df.index.name = "step"
    df.to_csv(RESULTS_DIR / "equity_curves.csv")

    # Save performance summary as JSON
    summary = {
        "performance":      results["performance"],
        "forecast_metrics": results["forecast_metrics"],
        "budget":           results["budget"],
        "n_steps":          results["n_steps"],
    }
    with open(RESULTS_DIR / "performance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {RESULTS_DIR}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget",          default=10_000.0, type=float)
    parser.add_argument("--risk_tolerance",  default="medium")
    parser.add_argument("--rebalance_every", default=1, type=int,
                        help="Rebalance every N steps (1=daily, 5=weekly, 21=monthly)")
    args = parser.parse_args()

    run_backtest(
        budget=args.budget,
        risk_tolerance=args.risk_tolerance,
        rebalance_every=args.rebalance_every,
    )
