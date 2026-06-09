"""
Training loop for the LSTM forecasting model.

Usage:
    python ml/training/train.py --pair USD_EUR --epochs 50
"""

import argparse
import logging
import random
import sys
from pathlib import Path

# Ensure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.models.lstm_model import LSTMForecaster
from ml.models.model_utils import save_model
from ml.training.evaluate import metrics_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR  = Path(__file__).resolve().parents[2] / "data" / "processed"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_tensors(pair_name: str) -> tuple:
    base = PROCESSED_DIR / pair_name
    X_train = torch.tensor(np.load(base / "X_train.npy"), dtype=torch.float32)
    y_train = torch.tensor(np.load(base / "y_train.npy"), dtype=torch.float32).unsqueeze(1)
    X_val   = torch.tensor(np.load(base / "X_val.npy"),   dtype=torch.float32)
    y_val   = torch.tensor(np.load(base / "y_val.npy"),   dtype=torch.float32).unsqueeze(1)
    X_test  = torch.tensor(np.load(base / "X_test.npy"),  dtype=torch.float32)
    y_test  = torch.tensor(np.load(base / "y_test.npy"),  dtype=torch.float32).unsqueeze(1)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    pair_name:   str,
    hidden_size: int   = 128,
    num_layers:  int   = 2,
    dropout:     float = 0.2,
    lr:          float = 1e-3,
    batch_size:  int   = 64,
    epochs:      int   = 50,
    patience:    int   = 10,
    seed:        int   = 42,
) -> dict:
    """
    Train the LSTM on one currency pair with early stopping.

    Returns:
        dict with best val_loss and test metrics
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(f"Training LSTM for {pair_name} on {DEVICE} (seed={seed})")

    X_train, y_train, X_val, y_val, X_test, y_test = load_tensors(pair_name)

    input_size = X_train.shape[2]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model     = LSTMForecaster(input_size, hidden_size, num_layers, dropout).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(X_train)

        # ── Validate ──
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(DEVICE))
            val_loss = criterion(val_pred, y_val.to(DEVICE)).item()

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:>3}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_model(model, pair_name, tag="best")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    # ── Test evaluation ──
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test.to(DEVICE)).cpu().numpy().squeeze()
        test_true = y_test.numpy().squeeze()

    report = metrics_report(pair_name, test_true, test_pred)
    report["model"]          = "LSTM"
    report["best_val_loss"]  = best_val_loss
    logger.info(f"  Test → RMSE={report['rmse']:.6f}  MAE={report['mae']:.6f}  DirAcc={report['directional_accuracy']:.3f}")
    return report


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_name",   default="EUR_USD",  help="Currency pair name")
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--num_layers",  default=2,   type=int)
    parser.add_argument("--dropout",     default=0.2, type=float)
    parser.add_argument("--lr",          default=1e-3, type=float)
    parser.add_argument("--batch_size",  default=64,  type=int)
    parser.add_argument("--epochs",      default=50,  type=int)
    parser.add_argument("--patience",    default=10,  type=int)
    parser.add_argument("--seed",        default=42,  type=int)
    args = parser.parse_args()

    result = train(**vars(args))
    print("\nFinal metrics:", result)
