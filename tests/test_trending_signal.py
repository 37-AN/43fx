"""Test that the strategy produces signals on synthetic trending data with relaxed mode."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

import pandas as pd

from src.data.synthetic_generator import generate_trending_series
from src.strategy.ema_trend_atr import EMATrendATRConfig, EMATrendATRStrategyLogic


class TrendingSignalTests(unittest.TestCase):
    """Validate that trade signals fire on controlled upward-trending data."""

    def setUp(self) -> None:
        self.config = EMATrendATRConfig(
            ema_fast_period=20,
            ema_slow_period=50,
            atr_period=14,
            atr_min_pips=7.0,
            pullback_lookback_min=3,
            pullback_lookback_max=5,
            swing_lookback=5,
            pip_size=0.0001,
            debug=False,
            relaxed_mode=True,  # relax filters for controlled validation
        )
        self.strategy = EMATrendATRStrategyLogic(self.config)
        self.session = {"start_hour": 0, "end_hour": 23, "timezone": "UTC"}

    def test_trending_data_produces_long_signal(self) -> None:
        """With relaxed_mode and trending data, at least 1 LONG signal must fire."""
        df = generate_trending_series(
            bars=3000,
            drift_per_bar=0.00005,
            volatility=0.0003,
            seed=42,
        )
        indicators = self.strategy.build_indicator_frame(df)

        signals_found: list[dict] = []
        min_warmup = max(self.config.ema_slow_period, self.config.atr_period) + 2

        for i in range(min_warmup, len(indicators)):
            window = indicators.iloc[: i + 1]
            row = window.iloc[-1]
            ts = window.index[-1]

            candle = {
                "timestamp": ts.to_pydatetime(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 1000)),
            }

            signal = self.strategy.generate_signal(
                candle,
                {"df": window},
                {"session": self.session},
            )
            if signal is not None:
                signals_found.append(signal)
                # One valid signal is sufficient to prove the path works
                break

        self.assertGreater(
            len(signals_found),
            0,
            "Expected at least 1 LONG signal on trending data with relaxed_mode=True. "
            f"Diagnostics: {self.strategy.diagnostics.summary()}",
        )
        self.assertEqual(signals_found[0]["direction"], "LONG")

    def test_diagnostics_counters_populated(self) -> None:
        """After running through bars, diagnostic counters should be non-zero."""
        df = generate_trending_series(bars=200, drift_per_bar=0.00005, volatility=0.0003, seed=99)
        indicators = self.strategy.build_indicator_frame(df)
        min_warmup = max(self.config.ema_slow_period, self.config.atr_period) + 2

        for i in range(min_warmup, len(indicators)):
            window = indicators.iloc[: i + 1]
            row = window.iloc[-1]
            ts = window.index[-1]

            candle = {
                "timestamp": ts.to_pydatetime(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": 1000.0,
            }
            self.strategy.generate_signal(candle, {"df": window}, {"session": self.session})

        diag = self.strategy.diagnostics.summary()
        self.assertGreater(diag["bars_evaluated"], 0, "bars_evaluated should be > 0")
        self.assertGreater(diag["session_pass_count"], 0, "session_pass should be > 0 in relaxed mode")

    def test_synthetic_generator_output_shape(self) -> None:
        """Verify synthetic generator produces valid OHLCV DataFrame."""
        df = generate_trending_series(bars=100, drift_per_bar=0.0001, volatility=0.0005)
        self.assertEqual(len(df), 100)
        for col in ["open", "high", "low", "close", "volume"]:
            self.assertIn(col, df.columns)
        # High should always be >= close and open
        self.assertTrue((df["high"] >= df["close"]).all())
        self.assertTrue((df["high"] >= df["open"]).all())
        # Low should always be <= close and open
        self.assertTrue((df["low"] <= df["close"]).all())
        self.assertTrue((df["low"] <= df["open"]).all())


if __name__ == "__main__":
    unittest.main()
