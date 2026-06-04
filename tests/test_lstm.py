"""
Unit tests for the LSTM model and evaluation utilities.
Run with: python -m pytest tests/test_lstm.py -v
"""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.models.lstm_model import LSTMForecaster
from ml.training.evaluate import rmse, mae, directional_accuracy, metrics_report


# ── LSTMForecaster ─────────────────────────────────────────────────────────────

class TestLSTMForecaster:

    def test_output_shape(self):
        model = LSTMForecaster(input_size=7, hidden_size=32, num_layers=1, dropout=0.0)
        x = torch.randn(16, 60, 7)   # batch=16, seq=60, features=7
        out = model(x)
        assert out.shape == (16, 1)

    def test_different_batch_sizes(self):
        model = LSTMForecaster(input_size=5, hidden_size=32, num_layers=2, dropout=0.1)
        for batch in [1, 8, 32]:
            x = torch.randn(batch, 30, 5)
            assert model(x).shape == (batch, 1)

    def test_forward_no_nan(self):
        model = LSTMForecaster(input_size=7)
        x = torch.randn(4, 60, 7)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_single_layer_no_dropout_error(self):
        """LSTM with 1 layer should not raise due to dropout on single layer."""
        model = LSTMForecaster(input_size=4, hidden_size=16, num_layers=1, dropout=0.3)
        x = torch.randn(2, 10, 4)
        out = model(x)
        assert out.shape == (2, 1)

    def test_gradients_flow(self):
        model = LSTMForecaster(input_size=7, hidden_size=32)
        x = torch.randn(4, 60, 7)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ── Evaluation metrics ─────────────────────────────────────────────────────────

class TestEvaluationMetrics:

    def test_rmse_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_mae_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_rmse_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        # errors: 1, 0, 1 → MSE=2/3 → RMSE=sqrt(2/3)
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(2 / 3))

    def test_directional_accuracy_perfect(self):
        y = np.array([1.0, 2.0, 3.0, 2.5])
        assert directional_accuracy(y, y) == pytest.approx(1.0)

    def test_metrics_report_keys(self):
        y = np.linspace(1, 2, 50)
        report = metrics_report("USD_EUR", y, y + 0.01)
        assert set(report.keys()) == {"pair", "rmse", "mae", "directional_accuracy"}
