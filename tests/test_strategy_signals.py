"""Strategy signal tests using synthetic candles."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest

import pandas as pd

from src.strategy.ema_trend_atr import EMATrendATRConfig, EMATrendATRStrategyLogic


class StrategySignalTests(unittest.TestCase):
    def setUp(self) -> None:
        config = EMATrendATRConfig(
            ema_fast_period=5,
            ema_slow_period=8,
            atr_period=5,
            atr_min_pips=1.0,
            pullback_lookback_min=3,
            pullback_lookback_max=5,
            swing_lookback=5,
            pip_size=0.0001,
        )
        self.strategy = EMATrendATRStrategyLogic(config)
        self.session = {"start_hour": 8, "end_hour": 17, "timezone": "UTC"}

    def _make_df_for_long_signal(self) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
        step = (1.1300 - 1.1000) / max(len(idx) - 1, 1)
        base = [1.1000 + i * step for i in range(len(idx))]

        close = pd.Series(base, index=idx)
        open_ = close.shift(1).fillna(close.iloc[0])
        high = pd.concat([open_, close], axis=1).max(axis=1) + 0.0008
        low = pd.concat([open_, close], axis=1).min(axis=1) - 0.0008

        df = pd.DataFrame(
            {
                "open": open_.values,
                "high": high.values,
                "low": low.values,
                "close": close.values,
                "volume": 1000,
            },
            index=idx,
        )

        # Force a recent pullback touch then reclaim above EMA.
        df.iloc[-4, df.columns.get_loc("low")] = df.iloc[-4]["close"] - 0.0030
        df.iloc[-1, df.columns.get_loc("close")] = df.iloc[-2]["close"] + 0.0015
        return df

    def test_generate_long_signal(self) -> None:
        df = self._make_df_for_long_signal()
        ind = self.strategy.build_indicator_frame(df)

        candle_time = datetime(2024, 1, 4, 10, 0, tzinfo=timezone.utc)
        candle = {
            "timestamp": candle_time,
            "open": float(df.iloc[-1]["open"]),
            "high": float(df.iloc[-1]["high"]),
            "low": float(df.iloc[-1]["low"]),
            "close": float(df.iloc[-1]["close"]),
            "volume": float(df.iloc[-1]["volume"]),
        }

        signal = self.strategy.generate_signal(candle, {"df": ind}, {"session": self.session})
        self.assertIsNotNone(signal)
        self.assertIn(signal["direction"], {"LONG", "SHORT"})

    def test_session_filter_blocks_signal(self) -> None:
        df = self._make_df_for_long_signal()
        ind = self.strategy.build_indicator_frame(df)

        candle_time = datetime(2024, 1, 4, 3, 0, tzinfo=timezone.utc)
        candle = {
            "timestamp": candle_time,
            "open": float(df.iloc[-1]["open"]),
            "high": float(df.iloc[-1]["high"]),
            "low": float(df.iloc[-1]["low"]),
            "close": float(df.iloc[-1]["close"]),
            "volume": float(df.iloc[-1]["volume"]),
        }

        signal = self.strategy.generate_signal(candle, {"df": ind}, {"session": self.session})
        self.assertIsNone(signal)


if __name__ == "__main__":
    unittest.main()
