"""Historical and backtest data loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
COLUMN_ALIASES = {
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe has standard OHLCV lowercase columns."""
    normalized = df.rename(columns=COLUMN_ALIASES).copy()
    lower_map = {column: column.lower() for column in normalized.columns}
    normalized = normalized.rename(columns=lower_map)

    missing = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return normalized[REQUIRED_COLUMNS]


def load_historical_csv(path: str | Path) -> pd.DataFrame:
    """Load historical candles from CSV with datetime index and OHLCV columns."""
    csv_path = Path(path)
    df = pd.read_csv(csv_path)

    datetime_column_candidates = ["datetime", "date", "time", "timestamp", "Datetime", "Date"]
    datetime_column: Optional[str] = next(
        (col for col in datetime_column_candidates if col in df.columns),
        None,
    )

    if datetime_column is None:
        raise ValueError("CSV must include one datetime column (e.g. datetime/date/timestamp).")

    df[datetime_column] = pd.to_datetime(df[datetime_column], utc=True, errors="coerce")
    df = df.dropna(subset=[datetime_column]).set_index(datetime_column).sort_index()

    standardized = standardize_columns(df)
    for column in REQUIRED_COLUMNS:
        standardized[column] = pd.to_numeric(standardized[column], errors="coerce")

    standardized = standardized.dropna()
    return standardized


def get_backtest_data(instrument: str, timeframe: str, data_dir: str | Path = "data") -> pd.DataFrame:
    """Load backtest data for instrument and timeframe from default naming convention."""
    file_name = f"{instrument}_{timeframe}.csv"
    file_path = Path(data_dir) / file_name
    return load_historical_csv(file_path)
