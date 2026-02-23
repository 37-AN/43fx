"""Unit tests for risk engine sizing and guardrails."""

from __future__ import annotations

import unittest

from src.risk.risk_engine import RiskEngine


class RiskEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RiskEngine(
            risk_per_trade=0.01,
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.12,
            pip_value_per_lot=10.0,
            pip_size=0.0001,
        )

    def test_size_position_standard_case(self) -> None:
        lots = self.engine.size_position(
            entry_price=1.1000,
            stop_loss_price=1.0950,
            equity=10_000,
            symbol="EURUSD",
        )
        self.assertAlmostEqual(lots, 0.2, places=4)

    def test_size_position_changes_with_sl_distance(self) -> None:
        wider_sl = self.engine.size_position(1.1000, 1.0900, equity=10_000)
        tighter_sl = self.engine.size_position(1.1000, 1.0980, equity=10_000)
        self.assertGreater(tighter_sl, wider_sl)

    def test_size_position_changes_with_equity(self) -> None:
        low_equity = self.engine.size_position(1.1000, 1.0950, equity=5_000)
        high_equity = self.engine.size_position(1.1000, 1.0950, equity=20_000)
        self.assertGreater(high_equity, low_equity)

    def test_guardrail_daily_loss(self) -> None:
        self.assertFalse(self.engine.can_open_new_trade(0.05, 0.01))

    def test_guardrail_drawdown(self) -> None:
        self.assertFalse(self.engine.can_open_new_trade(0.01, 0.15))


if __name__ == "__main__":
    unittest.main()
