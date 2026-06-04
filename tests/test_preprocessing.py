"""
Unit tests for the data pipeline.
Run with: pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.data_pipeline.preprocessing import (
    clean,
    add_features,
    normalize,
    split,
    make_sequences,
)


@pytest.fixture
def sample_df():
    """Minimal synthetic OHLC DataFrame."""
    n = 200
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 1.10 + np.cumsum(rng.normal(0, 0.002, n))
    return pd.DataFrame({
        "Date":  dates,
        "Open":  close - rng.uniform(0, 0.003, n),
        "High":  close + rng.uniform(0, 0.004, n),
        "Low":   close - rng.uniform(0, 0.004, n),
        "Close": close,
    })


def test_clean_forward_fills(sample_df):
    df = sample_df.copy()
    df.loc[5, "Close"] = np.nan
    cleaned = clean(df)
    assert cleaned["Close"].isna().sum() == 0


def test_clean_sorts_by_date(sample_df):
    shuffled = sample_df.sample(frac=1, random_state=0)
    cleaned = clean(shuffled)
    assert cleaned["Date"].is_monotonic_increasing


def test_add_features_columns(sample_df):
    df = clean(sample_df)
    df = add_features(df)
    assert "log_return" in df.columns
    assert "rolling_vol_30" in df.columns
    assert "rolling_vol_60" in df.columns
    assert df.isna().sum().sum() == 0


def test_normalize_range(sample_df):
    df = clean(sample_df)
    df = add_features(df)
    scaled, scaler = normalize(df)
    for col in ["Close", "log_return"]:
        if col in scaled.columns:
            assert scaled[col].min() >= -1e-6
            assert scaled[col].max() <= 1 + 1e-6


def test_split_sizes(sample_df):
    df = clean(sample_df)
    df = add_features(df)
    train, val, test = split(df, train_ratio=0.8, val_ratio=0.1)
    total = len(train) + len(val) + len(test)
    assert total == len(df)
    assert len(train) > len(val)
    assert len(val) >= len(test)


def test_make_sequences_shape(sample_df):
    df = clean(sample_df)
    df = add_features(df)
    scaled, _ = normalize(df)
    feature_cols = ["Open", "High", "Low", "Close", "log_return"]
    seq_len = 30
    X, y = make_sequences(scaled, feature_cols, sequence_length=seq_len)
    assert X.shape == (len(scaled) - seq_len, seq_len, len(feature_cols))
    assert y.shape == (len(scaled) - seq_len,)
