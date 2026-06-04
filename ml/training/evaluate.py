"""
Evaluation metrics and reporting utilities.
"""

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of timesteps where predicted direction matches actual direction."""
    actual_dir    = np.sign(np.diff(y_true))
    predicted_dir = np.sign(np.diff(y_pred))
    return float(np.mean(actual_dir == predicted_dir))


def metrics_report(pair_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "pair":                 pair_name,
        "rmse":                 rmse(y_true, y_pred),
        "mae":                  mae(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Pretty-print a comparison table of model results."""
    df = pd.DataFrame(results)
    df = df.sort_values(["pair", "rmse"]).reset_index(drop=True)
    return df


# ── Portfolio-level metrics ────────────────────────────────────────────────────

def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Annualised Sharpe ratio from daily returns.
    risk_free_rate: annual rate (e.g. 0.02 = 2%)
    """
    daily_rf = risk_free_rate / 252
    excess   = returns - daily_rf
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Maximum peak-to-trough drawdown as a fraction (e.g. -0.12 = -12%).
    """
    peak    = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / (peak + 1e-8)
    return float(drawdown.min())


def total_return(equity_curve: np.ndarray) -> float:
    """Total return fraction from start to end of equity curve."""
    return float((equity_curve[-1] - equity_curve[0]) / (equity_curve[0] + 1e-8))


def portfolio_metrics(equity_curve: np.ndarray, risk_free_rate: float = 0.02) -> dict:
    """
    Compute all portfolio-level metrics from an equity curve (daily portfolio values).
    """
    daily_returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-8)
    return {
        "total_return_pct":    round(total_return(equity_curve) * 100, 4),
        "sharpe_ratio":        round(sharpe_ratio(daily_returns, risk_free_rate), 4),
        "max_drawdown_pct":    round(max_drawdown(equity_curve) * 100, 4),
        "annualised_vol_pct":  round(float(daily_returns.std() * np.sqrt(252)) * 100, 4),
        "final_value":         round(float(equity_curve[-1]), 2),
        "initial_value":       round(float(equity_curve[0]), 2),
    }
