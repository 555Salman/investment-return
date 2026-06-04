"""
Risk manager: computes volatility metrics from processed data
and enforces per-currency risk constraints for the LP optimizer.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def compute_volatility(pair_name: str, window: int = 30) -> dict:
    """
    Load the test split for a pair and compute rolling volatility stats.

    Returns:
        dict with keys: pair, mean_vol, max_vol, latest_vol, annualised_vol
    """
    path = PROCESSED_DIR / pair_name / "test.csv"
    df   = pd.read_csv(path)

    if "rolling_vol_30" not in df.columns:
        raise KeyError(f"'rolling_vol_30' not found in {path}. Run preprocessing first.")

    vol = df["rolling_vol_30"].dropna()

    return {
        "pair":          pair_name,
        "mean_vol":      float(vol.mean()),
        "max_vol":       float(vol.max()),
        "latest_vol":    float(vol.iloc[-1]),
        "annualised_vol": float(vol.mean() * np.sqrt(252)),
    }


def get_risk_limits(
    pairs: list[str],
    risk_tolerance: str = "medium",
) -> dict[str, float]:
    """
    Compute per-pair volatility limits to use as LP constraints.

    risk_tolerance:
        "low"    → limit = mean_vol * 1.0  (tight)
        "medium" → limit = mean_vol * 1.5
        "high"   → limit = mean_vol * 2.5  (loose)

    Returns:
        dict mapping pair_name → max allowed volatility weight
    """
    multipliers = {"low": 1.0, "medium": 1.5, "high": 2.5}
    mult = multipliers.get(risk_tolerance, 1.5)

    limits = {}
    for pair in pairs:
        stats = compute_volatility(pair)
        limits[pair] = stats["mean_vol"] * mult
        logger.info(
            f"  {pair}: mean_vol={stats['mean_vol']:.6f}  "
            f"limit={limits[pair]:.6f}  annualised={stats['annualised_vol']:.4f}"
        )
    return limits
