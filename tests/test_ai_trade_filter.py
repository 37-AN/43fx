"""Tests for AI trade filter training and inference gating."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone

import pandas as pd

from src.ai.trade_filter_model import extract_training_dataset, train_model
from src.strategy.ema_trend_atr import EMATrendATRConfig, EMATrendATRStrategyLogic


class _FakeModel:
    """Simple fake model for deterministic predict_proba behavior."""

    def __init__(self, positive_prob: float):
        self.positive_prob = positive_prob

    def predict_proba(self, X):  # noqa: N803
        return [[1.0 - self.positive_prob, self.positive_prob] for _ in X]


class AITradeFilterTests(unittest.TestCase):
    def _ai_config(self) -> dict:
        return {
            "ai": {
                "enabled": True,
                "model_type": "logistic",
                "probability_threshold": 0.55,
                "features": {
                    "use_session_flags": True,
                    "use_volatility_features": True,
                    "use_trend_slope": True,
                },
            }
        }

    def _build_signal_frame(self) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC")
        base = [1.1000 + i * 0.0002 for i in range(len(idx))]

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

        # Inject pullback then reclaim to trigger LONG setup.
        df.iloc[-4, df.columns.get_loc("low")] = df.iloc[-4]["close"] - 0.0030
        df.iloc[-1, df.columns.get_loc("close")] = df.iloc[-2]["close"] + 0.0015
        return df

    def test_model_training_on_synthetic_trades(self) -> None:
        rows = []
        for i in range(50):
            label_positive = i % 2 == 0
            rows.append(
                {
                    "entry_price": 1.1000 + i * 0.0001,
                    "stop_loss": 1.0980 + i * 0.0001,
                    "direction": "LONG" if i % 3 else "SHORT",
                    "atr_pips": 8.0 + (i % 5),
                    "ema_fast": 1.1000 + i * 0.00008,
                    "ema_slow": 1.0998 + i * 0.00006,
                    "ema_fast_prev": 1.0999 + i * 0.00008,
                    "ema_slow_prev": 1.0997 + i * 0.00006,
                    "recent_atr_percentile": 0.2 + (i % 10) / 10,
                    "entry_hour": 8 + (i % 10),
                    "pip_size": 0.0001,
                    "r_multiple": 1.8 if label_positive else -1.0,
                }
            )

        df = pd.DataFrame(rows)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            df.to_csv(tmp.name, index=False)
            csv_path = tmp.name

        X, y = extract_training_dataset(csv_path, self._ai_config())
        model = train_model(X, y, self._ai_config())

        self.assertIsNotNone(model)
        pred = model.predict(X[:1])
        self.assertEqual(len(pred), 1)

    def test_ai_filter_blocks_low_probability_trades(self) -> None:
        strategy_cfg = EMATrendATRConfig(
            ema_fast_period=20,
            ema_slow_period=50,
            atr_period=14,
            atr_min_pips=7.0,
            pullback_lookback_min=3,
            pullback_lookback_max=5,
            swing_lookback=5,
            pip_size=0.0001,
            ai_enabled=True,
        )
        session_cfg = {"start_hour": 8, "end_hour": 17, "timezone": "UTC"}

        candles = self._build_signal_frame()
        candle_time = datetime(2024, 1, 5, 10, 0, tzinfo=timezone.utc)
        candle = {
            "timestamp": candle_time,
            "open": float(candles.iloc[-1]["open"]),
            "high": float(candles.iloc[-1]["high"]),
            "low": float(candles.iloc[-1]["low"]),
            "close": float(candles.iloc[-1]["close"]),
            "volume": float(candles.iloc[-1]["volume"]),
        }

        high_prob_strategy = EMATrendATRStrategyLogic(
            strategy_cfg,
            ai_config=self._ai_config()["ai"],
            ai_model=_FakeModel(positive_prob=0.70),
        )
        indicators_high = high_prob_strategy.build_indicator_frame(candles)
        approved_signal = high_prob_strategy.generate_signal(
            candle,
            {"df": indicators_high},
            {"session": session_cfg},
        )

        low_prob_strategy = EMATrendATRStrategyLogic(
            strategy_cfg,
            ai_config=self._ai_config()["ai"],
            ai_model=_FakeModel(positive_prob=0.20),
        )
        indicators_low = low_prob_strategy.build_indicator_frame(candles)
        rejected_signal = low_prob_strategy.generate_signal(
            candle,
            {"df": indicators_low},
            {"session": session_cfg},
        )

        self.assertIsNotNone(approved_signal)
        self.assertIsNone(rejected_signal)


if __name__ == "__main__":
    unittest.main()
