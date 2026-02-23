"""Risk management and position sizing logic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskEngine:
    """Reusable risk engine for backtest and live trading."""

    risk_per_trade: float
    max_daily_loss_pct: float
    max_drawdown_pct: float
    pip_value_per_lot: float
    account_currency: str = "USD"
    pip_size: float = 0.0001

    def size_position(
        self,
        entry_price: float,
        stop_loss_price: float,
        equity: float,
        symbol: str = "EURUSD",
    ) -> float:
        """Compute position size in standard lots using fixed fractional risk.

        Intended use:
        - Backtest: call before creating each order.
        - Live: call after broker equity is fetched and signal SL is known.
        """
        if equity <= 0:
            return 0.0

        sl_distance = abs(entry_price - stop_loss_price)
        if sl_distance <= 0:
            return 0.0

        sl_pips = sl_distance / self.pip_size
        if sl_pips <= 0:
            return 0.0

        dollar_risk = equity * self.risk_per_trade
        lots = dollar_risk / (sl_pips * self.pip_value_per_lot)
        return max(round(lots, 4), 0.0)

    def can_open_new_trade(
        self,
        daily_realized_loss_pct: float,
        current_drawdown_pct: float,
    ) -> bool:
        """Check whether portfolio-level risk guardrails permit new exposure."""
        if daily_realized_loss_pct >= self.max_daily_loss_pct:
            return False
        if current_drawdown_pct >= self.max_drawdown_pct:
            return False
        return True
