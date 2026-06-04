"""
Helpers for saving and loading model weights and scalers.
"""

import pickle
from pathlib import Path

import torch
from sklearn.preprocessing import MinMaxScaler

CHECKPOINTS_DIR = Path(__file__).resolve().parents[2] / "data" / "checkpoints"


def save_model(model: torch.nn.Module, pair_name: str, tag: str = "best") -> Path:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINTS_DIR / f"{pair_name}_{tag}.pt"
    torch.save(model.state_dict(), path)
    return path


def load_model(model: torch.nn.Module, pair_name: str, tag: str = "best") -> torch.nn.Module:
    path = CHECKPOINTS_DIR / f"{pair_name}_{tag}.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def save_scaler(scaler: MinMaxScaler, pair_name: str) -> Path:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINTS_DIR / f"{pair_name}_scaler.pkl"
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    return path


def load_scaler(pair_name: str) -> MinMaxScaler:
    path = CHECKPOINTS_DIR / f"{pair_name}_scaler.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
