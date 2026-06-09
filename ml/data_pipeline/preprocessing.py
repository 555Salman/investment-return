"""
Preprocessing module.
Cleans raw exchange rate data, computes log returns,
applies normalization, and produces model-ready sequences.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


# ── Cleaning ───────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by date, forward-fill missing values, and drop any remaining NaNs.
    """
    df = df.sort_values("Date").reset_index(drop=True)

    missing_before = df.isnull().sum().sum()
    df = df.ffill()          # forward fill (carries last known value forward)
    df = df.dropna()         # drop rows that are still NaN (e.g., leading rows)
    missing_after  = df.isnull().sum().sum()

    if missing_before:
        logger.info(f"  Forward-filled {missing_before} missing values → {missing_after} remaining")

    return df


# ── Feature engineering ────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features used in modelling:
      - log_return      : daily log return of Close price
      - rolling_vol_30  : 30-day rolling volatility (std of log returns)
      - rolling_vol_60  : 60-day rolling volatility
    """
    df = df.copy()
    df["log_return"]     = np.log(df["Close"] / df["Close"].shift(1))
    df["rolling_vol_30"] = df["log_return"].rolling(window=30).std()
    df["rolling_vol_60"] = df["log_return"].rolling(window=60).std()

    # Drop the first 60 rows where rolling features are NaN
    df = df.dropna().reset_index(drop=True)
    return df


# ── Normalisation ──────────────────────────────────────────────────────────────

FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "log_return", "rolling_vol_30", "rolling_vol_60"]

def normalize(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLUMNS,
    scaler: MinMaxScaler | None = None,
) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Apply MinMax scaling to the specified feature columns.

    Args:
        df:           Cleaned DataFrame with feature columns present
        feature_cols: Columns to scale
        scaler:       Pre-fitted scaler (use when transforming val/test sets)

    Returns:
        (scaled_df, fitted_scaler)
    """
    df = df.copy()
    cols = [c for c in feature_cols if c in df.columns]

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])

    return df, scaler


# ── Train / val / test split ───────────────────────────────────────────────────

def split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float   = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split into train, validation, and test sets.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].reset_index(drop=True)
    val   = df.iloc[train_end:val_end].reset_index(drop=True)
    test  = df.iloc[val_end:].reset_index(drop=True)

    logger.info(f"  Split → train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ── Sequence builder for LSTM ──────────────────────────────────────────────────

def make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Close",
    sequence_length: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build overlapping sliding-window sequences for LSTM training.

    Args:
        df:              Scaled DataFrame
        feature_cols:    Input feature columns
        target_col:      Column to predict (next time step)
        sequence_length: Number of past days used as context (lookback window)

    Returns:
        X : shape (samples, sequence_length, num_features)
        y : shape (samples,)   — next-day Close value
    """
    features = df[feature_cols].values
    target   = df[target_col].values

    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(features[i - sequence_length : i])
        y.append(target[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    df: pd.DataFrame,
    pair_name: str,
    sequence_length: int = 60,
    save: bool = True,
) -> dict:
    """
    Run the complete preprocessing pipeline for one currency pair.

    Returns a dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test,
        scaler, train_df, val_df, test_df
    """
    logger.info(f"Preprocessing {pair_name}")

    df = clean(df)

    # Split on raw data BEFORE feature engineering so rolling stats for val/test
    # are never computed using data that belongs to a later split.
    # Each split uses a 59-row lookback buffer from the prior split to warm up
    # rolling_vol_60 (which requires 60 rows); those buffer rows carry NaN for
    # the first 59 positions and are removed by the dropna() inside add_features.
    _BUFFER = sequence_length - 1   # 59 rows: fills the rolling warm-up window

    train_raw, val_raw, test_raw = split(df)

    train_df = add_features(train_raw)

    _val_buf = pd.concat([train_raw.tail(_BUFFER), val_raw], ignore_index=True)
    val_df   = add_features(_val_buf)   # buffer rows (NaN rolling stats) auto-dropped

    _test_buf = pd.concat([val_raw.tail(_BUFFER), test_raw], ignore_index=True)
    test_df   = add_features(_test_buf)

    # Fit scaler on train only, transform all splits
    train_df, scaler = normalize(train_df)
    val_df,   _      = normalize(val_df,  scaler=scaler)
    test_df,  _      = normalize(test_df, scaler=scaler)

    feature_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns]

    X_train, y_train = make_sequences(train_df, feature_cols, sequence_length=sequence_length)
    X_val,   y_val   = make_sequences(val_df,   feature_cols, sequence_length=sequence_length)
    X_test,  y_test  = make_sequences(test_df,  feature_cols, sequence_length=sequence_length)

    logger.info(
        f"  Sequences → train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"
    )

    if save:
        import pickle
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out = PROCESSED_DATA_DIR / pair_name
        out.mkdir(exist_ok=True)
        np.save(out / "X_train.npy", X_train)
        np.save(out / "y_train.npy", y_train)
        np.save(out / "X_val.npy",   X_val)
        np.save(out / "y_val.npy",   y_val)
        np.save(out / "X_test.npy",  X_test)
        np.save(out / "y_test.npy",  y_test)
        train_df.to_csv(out / "train.csv", index=False)
        val_df.to_csv(  out / "val.csv",   index=False)
        test_df.to_csv( out / "test.csv",  index=False)

        # Save scaler so predictions can be inverse-transformed later
        with open(out / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"  Saved processed data → {out}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler":  scaler,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
    }
