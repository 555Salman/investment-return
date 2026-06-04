"""
Data ingestion module.
Downloads historical OHLC exchange rate data from Yahoo Finance
for the three currency pairs used in the research.
"""

import os
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

CURRENCY_PAIRS = {
    "EUR_USD": "EURUSD=X",
    "AUD_USD": "AUDUSD=X",
    "NZD_USD": "NZDUSD=X",
}

START_DATE = "2020-01-01"
END_DATE   = "2025-12-31"

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


# ── Download ───────────────────────────────────────────────────────────────────

def download_pair(ticker: str, pair_name: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLC data for a single currency pair from Yahoo Finance.

    Args:
        ticker:    Yahoo Finance ticker symbol (e.g. "USDEUR=X")
        pair_name: Human-readable name used for logging (e.g. "USD_EUR")
        start:     Start date string "YYYY-MM-DD"
        end:       End date string "YYYY-MM-DD"

    Returns:
        DataFrame with columns [Date, Open, High, Low, Close, Volume]
    """
    logger.info(f"Downloading {pair_name} ({ticker})  {start} → {end}")

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker symbol or date range.")

    # yfinance returns a MultiIndex when multiple tickers are requested;
    # for a single ticker the columns are already flat.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    df = df.reset_index()

    # Keep only the columns we need
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep]

    logger.info(f"  → {len(df)} rows  |  {df['Date'].min().date()} to {df['Date'].max().date()}")
    return df


def download_all(
    pairs: dict[str, str] | None = None,
    start: str = START_DATE,
    end: str = END_DATE,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Download data for all configured currency pairs.

    Args:
        pairs:  Dict mapping name → Yahoo ticker. Defaults to CURRENCY_PAIRS.
        start:  Start date "YYYY-MM-DD"
        end:    End date   "YYYY-MM-DD"
        save:   If True, save each DataFrame as a CSV in data/raw/

    Returns:
        Dict mapping pair name → DataFrame
    """
    if pairs is None:
        pairs = CURRENCY_PAIRS

    if save:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}

    for name, ticker in pairs.items():
        try:
            df = download_pair(ticker, name, start, end)
            results[name] = df

            if save:
                path = RAW_DATA_DIR / f"{name}.csv"
                df.to_csv(path, index=False)
                logger.info(f"  Saved → {path}")

        except Exception as exc:
            logger.error(f"Failed to download {name}: {exc}")

    logger.info(f"Download complete. {len(results)}/{len(pairs)} pairs fetched successfully.")
    return results


# ── Load from disk ─────────────────────────────────────────────────────────────

def load_pair(pair_name: str, data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Load a previously saved raw CSV for a currency pair.

    Args:
        pair_name: e.g. "USD_EUR"
        data_dir:  Directory containing the CSVs

    Returns:
        DataFrame with Date parsed as datetime
    """
    path = data_dir / f"{pair_name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run download_all() first."
        )

    df = pd.read_csv(path, parse_dates=["Date"])
    logger.info(f"Loaded {pair_name}: {len(df)} rows from {path}")
    return df


def load_all(data_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load all available raw CSVs from disk."""
    data: dict[str, pd.DataFrame] = {}
    for name in CURRENCY_PAIRS:
        try:
            data[name] = load_pair(name, data_dir)
        except FileNotFoundError as exc:
            logger.warning(str(exc))
    return data


# ── Quick summary ──────────────────────────────────────────────────────────────

def summarize(data: dict[str, pd.DataFrame]) -> None:
    """Print a quick overview of each downloaded dataset."""
    print("\n" + "=" * 60)
    print(f"{'Pair':<12} {'Rows':>6}  {'Start':>12}  {'End':>12}  {'Missing Close':>14}")
    print("=" * 60)
    for name, df in data.items():
        missing = df["Close"].isna().sum()
        print(
            f"{name:<12} {len(df):>6}  "
            f"{str(df['Date'].min().date()):>12}  "
            f"{str(df['Date'].max().date()):>12}  "
            f"{missing:>14}"
        )
    print("=" * 60 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = download_all()
    summarize(data)
