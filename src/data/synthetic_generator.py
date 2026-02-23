"""Synthetic price data generators for backtest validation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_trending_series(
    bars: int = 3000,
    drift_per_bar: float = 0.00005,
    volatility: float = 0.0003,
    seed: int = 42,
    start_price: float = 1.1000,
    start_date: str = "2024-01-02",
) -> pd.DataFrame:
    """Generate an upward-trending H1 OHLCV price series with controlled noise.

    Parameters
    ----------
    bars : int
        Number of hourly bars to generate.
    drift_per_bar : float
        Deterministic price increment per bar (controls trend strength).
    volatility : float
        Standard deviation of Gaussian noise added per bar.
    seed : int
        Random seed for reproducibility.
    start_price : float
        Opening price of the first bar.
    start_date : str
        ISO date string for the first bar timestamp.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and ``open``, ``high``, ``low``,
        ``close``, ``volume`` columns.
    """
    rng = np.random.default_rng(seed)

    idx = pd.date_range(start_date, periods=bars, freq="h", tz="UTC")

    # Build close prices: deterministic drift + noise
    noise = rng.normal(0, volatility, bars)
    cumulative = np.cumsum(drift_per_bar + noise)
    closes = start_price + cumulative

    # Derive open from previous close (first bar opens at start_price)
    opens = np.empty(bars)
    opens[0] = start_price
    opens[1:] = closes[:-1]

    # High/low: ensure they bracket open/close and add realistic wicks
    wick_up = np.abs(rng.normal(0, volatility * 0.8, bars)) + 0.0001
    wick_dn = np.abs(rng.normal(0, volatility * 0.8, bars)) + 0.0001

    highs = np.maximum(opens, closes) + wick_up
    lows = np.minimum(opens, closes) - wick_dn

    volumes = rng.integers(500, 5000, size=bars)

    df = pd.DataFrame(
        {
            "open": np.round(opens, 5),
            "high": np.round(highs, 5),
            "low": np.round(lows, 5),
            "close": np.round(closes, 5),
            "volume": volumes,
        },
        index=idx,
    )
    df.index.name = "datetime"
    return df
