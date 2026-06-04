"""
Forecast Agent.
Triggered by the orchestrator to run LSTM inference on the latest data
and return next-step price forecasts for each currency pair.
"""

import sys
from pathlib import Path

import pickle

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.agents.base_agent import BaseAgent, AgentStatus
from ml.models.lstm_model import LSTMForecaster
from ml.models.model_utils import load_model

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PAIRS         = ["EUR_USD", "AUD_USD", "NZD_USD"]

# Close is the 4th column (index 3) in the scaler's feature set
# ["Open","High","Low","Close","log_return","rolling_vol_30","rolling_vol_60"]
_CLOSE_IDX = 3
_N_FEATURES = 7


def _inverse_close(scaler, scaled_vals: np.ndarray) -> np.ndarray:
    """Inverse-transform an array of MinMax-scaled Close prices."""
    dummy = np.zeros((len(scaled_vals), _N_FEATURES))
    dummy[:, _CLOSE_IDX] = scaled_vals
    return scaler.inverse_transform(dummy)[:, _CLOSE_IDX]


class ForecastAgent(BaseAgent):
    """
    Loads saved LSTM models and runs inference to produce exchange rate forecasts.
    """

    def __init__(self):
        super().__init__("ForecastAgent")
        self._models: dict[str, LSTMForecaster] = {}

    def _load_models(self) -> None:
        """Load best saved weights for each pair (cached after first load)."""
        for pair in PAIRS:
            if pair in self._models:
                continue
            try:
                X_sample = np.load(PROCESSED_DIR / pair / "X_test.npy")
                input_size = X_sample.shape[2]
                model = LSTMForecaster(input_size=input_size)
                model = load_model(model, pair, tag="best")
                self._models[pair] = model
                self._record("model_loaded", {"pair": pair})
            except FileNotFoundError:
                self._record("model_missing", {"pair": pair,
                    "hint": f"Run: python ml/training/train.py --pair_name {pair}"})

    def _load_scaler(self, pair: str):
        """Load the MinMaxScaler saved during preprocessing, or None if missing."""
        path = PROCESSED_DIR / pair / "scaler.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def forecast_pair(self, pair: str) -> dict | None:
        """
        Run inference on the last 60 days of the test set for a single pair.

        Returns:
            dict with keys: pair, predicted_next, last_actual, forecast_return
            Values are real (inverse-transformed) prices when scaler is available.
        """
        if pair not in self._models:
            return None

        X_test = np.load(PROCESSED_DIR / pair / "X_test.npy")
        y_test = np.load(PROCESSED_DIR / pair / "y_test.npy")

        # Use the last window in the test set as "current market state"
        last_window = torch.tensor(X_test[-1:], dtype=torch.float32)  # (1, 60, features)

        model = self._models[pair]
        model.eval()
        with torch.no_grad():
            pred_scaled = float(model(last_window).item())

        last_actual_scaled = float(y_test[-1])

        scaler = self._load_scaler(pair)
        if scaler is not None:
            pred        = float(_inverse_close(scaler, np.array([pred_scaled]))[0])
            last_actual = float(_inverse_close(scaler, np.array([last_actual_scaled]))[0])
        else:
            pred        = pred_scaled
            last_actual = last_actual_scaled

        forecast_return = (pred - last_actual) / (last_actual + 1e-8)

        return {
            "pair":             pair,
            "predicted_next":   pred,
            "last_actual":      last_actual,
            "forecast_return":  forecast_return,
        }

    async def run(self) -> dict[str, dict]:
        """
        Load all models and return forecasts for all pairs.

        Returns:
            dict mapping pair_name → forecast dict
        """
        self.status = AgentStatus.RUNNING
        self._record("forecasting_started")
        self._load_models()

        forecasts = {}
        for pair in PAIRS:
            result = self.forecast_pair(pair)
            if result:
                forecasts[pair] = result
                self._record("forecast_ready", {
                    "pair": pair,
                    "forecast_return": f"{result['forecast_return']*100:.3f}%"
                })
            else:
                self._record("forecast_skipped", {"pair": pair})

        self.status = AgentStatus.IDLE
        return forecasts
