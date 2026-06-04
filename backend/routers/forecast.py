"""
Forecast endpoints.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pickle

import numpy as np
import torch
from fastapi import APIRouter, HTTPException

from backend.services.forecast_service import forecast_service
from ml.models.lstm_model  import LSTMForecaster
from ml.models.model_utils import load_model

router = APIRouter(prefix="/api/forecast", tags=["Forecast"])

VALID_PAIRS   = {"EUR_USD", "AUD_USD", "NZD_USD"}
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


@router.get("/", summary="Get forecasts for all currency pairs")
async def get_all_forecasts():
    """
    Returns next-step LSTM forecasts for EUR/USD, AUD/USD, NZD/USD.
    Models must be trained before calling this endpoint.
    """
    result = await forecast_service.get_all_forecasts()
    if not result["forecasts"]:
        raise HTTPException(
            status_code=503,
            detail="No trained models found. Run: python ml/training/train.py --pair_name EUR_USD"
        )
    return result


@router.get("/{pair}", summary="Get forecast for a single currency pair")
async def get_pair_forecast(pair: str):
    """
    Returns the LSTM forecast for a specific pair.
    Pair format: EUR_USD | AUD_USD | NZD_USD
    """
    if pair not in VALID_PAIRS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid pair '{pair}'. Choose from: {sorted(VALID_PAIRS)}"
        )
    result = await forecast_service.get_pair_forecast(pair)
    if result is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model for {pair} not found. Train it first."
        )
    return result


@router.get("/{pair}/history", summary="Full actual vs predicted series for the test set")
def get_pair_history(pair: str):
    """
    Returns the complete test-set actual prices and LSTM predictions
    so the frontend can draw a proper historical chart.
    """
    if pair not in VALID_PAIRS:
        raise HTTPException(status_code=400, detail=f"Invalid pair '{pair}'.")

    base = PROCESSED_DIR / pair
    try:
        X_test = np.load(base / "X_test.npy")
        y_test = np.load(base / "y_test.npy")
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail=f"Processed data for {pair} not found.")

    try:
        model = LSTMForecaster(input_size=X_test.shape[2])
        model = load_model(model, pair, tag="best")
        model.eval()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail=f"Model for {pair} not trained yet.")

    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy().squeeze()

    # Inverse-transform scaled values back to real exchange rates
    scaler_path = base / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        # Close is index 3 in ["Open","High","Low","Close","log_return","rolling_vol_30","rolling_vol_60"]
        close_idx  = 3
        n_features = 7

        def inv_close(vals: np.ndarray) -> np.ndarray:
            dummy = np.zeros((len(vals), n_features))
            dummy[:, close_idx] = vals
            return scaler.inverse_transform(dummy)[:, close_idx]

        actual    = inv_close(y_test)
        predicted = inv_close(preds if preds.ndim == 1 else preds.flatten())
    else:
        actual    = y_test
        predicted = preds.flatten() if preds.ndim > 1 else preds

    return {
        "pair":      pair,
        "actual":    [round(float(v), 6) for v in actual],
        "predicted": [round(float(v), 6) for v in predicted],
    }
