"""
PyTorch Dataset wrappers for the preprocessed multicurrency time-series data.

Typical use:
    from ml.data_pipeline.dataset import MultiCurrencyDataset, load_pair_datasets

    train_ds, val_ds, test_ds = load_pair_datasets("EUR_USD")
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


class MultiCurrencyDataset(Dataset):
    """
    Wraps the X/y .npy arrays produced by preprocessing.run_pipeline().

    Args:
        X: shape (N, sequence_length, n_features) — input sequences
        y: shape (N,) — next-day Close targets
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_pair_datasets(
    pair_name: str,
    processed_dir: Path = PROCESSED_DIR,
) -> tuple["MultiCurrencyDataset", "MultiCurrencyDataset", "MultiCurrencyDataset"]:
    """
    Load train / val / test datasets for a single currency pair.

    Returns:
        (train_dataset, val_dataset, test_dataset)

    Raises:
        FileNotFoundError: if preprocessed arrays are missing — run
            ``python ml/data_pipeline/preprocessing.py`` first.
    """
    base = processed_dir / pair_name
    splits = {}
    for split in ("train", "val", "test"):
        x_path = base / f"X_{split}.npy"
        y_path = base / f"y_{split}.npy"
        if not x_path.exists():
            raise FileNotFoundError(
                f"Preprocessed arrays not found at {base}. "
                "Run: python ml/data_pipeline/preprocessing.py"
            )
        splits[split] = MultiCurrencyDataset(
            np.load(x_path),
            np.load(y_path),
        )

    return splits["train"], splits["val"], splits["test"]


def load_all_pairs(
    pairs: list[str] | None = None,
    processed_dir: Path = PROCESSED_DIR,
) -> dict[str, dict[str, "MultiCurrencyDataset"]]:
    """
    Load datasets for all (or specified) pairs.

    Returns:
        {pair_name: {"train": ds, "val": ds, "test": ds}}
    """
    if pairs is None:
        pairs = [p.name for p in processed_dir.iterdir() if p.is_dir()]
        pairs = sorted(pairs)

    result = {}
    for pair in pairs:
        try:
            train, val, test = load_pair_datasets(pair, processed_dir)
            result[pair] = {"train": train, "val": val, "test": test}
        except FileNotFoundError:
            pass  # pair not yet preprocessed — skip silently
    return result
