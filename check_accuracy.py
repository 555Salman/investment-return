"""
Compute and report accuracy metrics for all trained LSTM models
and run the backtest to get portfolio-level metrics.
"""
import json, pickle
import numpy as np
import torch
from pathlib import Path

from ml.models.lstm_model import LSTMForecaster
from ml.models.model_utils import load_model
from ml.training.evaluate import metrics_report, compare_models, portfolio_metrics
from ml.models.arima_baseline import fit_arima, rolling_forecast, evaluate_arima
from ml.training.backtest import run_backtest

PAIRS = ['EUR_USD', 'AUD_USD', 'NZD_USD']
PROCESSED = Path('data/processed')

print("=" * 65)
print("  LSTM MODEL ACCURACY  (test set — real exchange-rate scale)")
print("=" * 65)

lstm_results = []
for pair in PAIRS:
    base = PROCESSED / pair
    X_test = np.load(base / 'X_test.npy')
    y_test = np.load(base / 'y_test.npy')

    model = LSTMForecaster(input_size=X_test.shape[2])
    model = load_model(model, pair, tag='best')
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy().squeeze()

    # Inverse-scale to real prices
    with open(base / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(base / 'feature_columns.json') as f:
        cols = json.load(f)
    ci = cols.index('Close')
    n  = scaler.n_features_in_

    def inv(vals):
        d = np.zeros((len(vals), n))
        d[:, ci] = vals
        return scaler.inverse_transform(d)[:, ci]

    y_real   = inv(y_test)
    p_real   = inv(preds.flatten())

    r = metrics_report(pair, y_real, p_real)
    lstm_results.append(r)

df = compare_models(lstm_results)
print(df.to_string(index=False))

# ── ARIMA baseline ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ARIMA BASELINE ACCURACY  (test set — real price scale)")
print("=" * 65)

for pair in PAIRS:
    try:
        r = evaluate_arima(pair, PROCESSED, order=(5, 1, 0))
        print(f"  {pair}: RMSE={r['rmse']:.5f}  MAE={r['mae']:.5f}")
    except Exception as e:
        print(f"  {pair}: ARIMA failed — {e}")

# ── Backtest / portfolio metrics ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PORTFOLIO BACKTEST  (walk-forward, $10,000 budget)")
print("=" * 65)
try:
    bt = run_backtest(budget=10_000.0, risk_tolerance='medium', rebalance_every=1)
    for strategy, equity in bt.items():
        eq = np.array(equity)
        pm = portfolio_metrics(eq)
        print(f"\n  Strategy: {strategy}")
        print(f"    Total Return    : {pm['total_return_pct']:+.2f}%")
        print(f"    Sharpe Ratio    : {pm['sharpe_ratio']:.4f}")
        print(f"    Max Drawdown    : {pm['max_drawdown_pct']:.2f}%")
        print(f"    Annualised Vol  : {pm['annualised_vol_pct']:.2f}%")
        print(f"    Final Value     : ${pm['final_value']:,.2f}")
except Exception as e:
    print(f"  Backtest failed: {e}")

print("\n" + "=" * 65)
print("  DONE")
print("=" * 65)
