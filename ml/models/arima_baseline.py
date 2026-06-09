"""
ARIMA baseline model for comparison against LSTM.
Fits one model per currency pair and evaluates on the test set.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from ml.training.evaluate import rmse, mae

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")   # suppress statsmodels convergence warnings


def check_stationarity(series: pd.Series, significance: float = 0.05) -> bool:
    """Augmented Dickey-Fuller test. Returns True if series is stationary."""
    result = adfuller(series.dropna())
    p_value = result[1]
    logger.info(f"  ADF p-value: {p_value:.4f} → {'stationary' if p_value < significance else 'non-stationary'}")
    return p_value < significance


def fit_arima(
    train: np.ndarray,
    order: tuple[int, int, int] = (5, 1, 0),
) -> ARIMA:
    """
    Fit an ARIMA model on training data.

    Args:
        train: 1-D array of close prices
        order: (p, d, q) ARIMA order

    Returns:
        Fitted ARIMA results object
    """
    model = ARIMA(train, order=order)
    result = model.fit()
    return result


def rolling_forecast(
    train: np.ndarray,
    test: np.ndarray,
    order: tuple[int, int, int] = (5, 1, 0),
) -> np.ndarray:
    """
    Walk-forward (rolling) ARIMA forecast over the test window.
    Refits the model on each step to incorporate new observations.

    Args:
        train: Training close prices
        test:  Test close prices
        order: ARIMA (p, d, q)

    Returns:
        predictions: Array of one-step-ahead forecasts, same length as test
    """
    history     = list(train)
    predictions = []

    for t in range(len(test)):
        model  = ARIMA(history, order=order)
        result = model.fit()
        yhat   = result.forecast(steps=1)[0]
        predictions.append(yhat)
        history.append(test[t])   # expand window with true value

        if (t + 1) % 50 == 0:
            logger.info(f"  ARIMA step {t + 1}/{len(test)}")

    return np.array(predictions)


def evaluate_arima(
    pair_name: str,
    processed_dir: Path,
    order: tuple[int, int, int] = (5, 1, 0),
) -> dict:
    """
    Load processed data for a currency pair, run rolling ARIMA,
    and return RMSE / MAE metrics.
    """
    data_path = processed_dir / pair_name
    train_df  = pd.read_csv(data_path / "train.csv")
    test_df   = pd.read_csv(data_path / "test.csv")

    train_close = train_df["Close"].values
    test_close  = test_df["Close"].values

    logger.info(f"Running ARIMA{order} for {pair_name}")
    check_stationarity(pd.Series(np.diff(np.log(np.maximum(train_close, 1e-8)))))

    preds = rolling_forecast(train_close, test_close, order=order)

    metrics = {
        "pair":  pair_name,
        "model": f"ARIMA{order}",
        "rmse":  rmse(test_close, preds),
        "mae":   mae(test_close, preds),
    }
    logger.info(f"  RMSE={metrics['rmse']:.6f}  MAE={metrics['mae']:.6f}")
    return metrics
