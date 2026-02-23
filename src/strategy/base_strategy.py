"""Base strategy interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional


class BaseSignalStrategy(ABC):
    """Abstract interface for candle-based signal generation."""

    @abstractmethod
    def generate_signal(
        self,
        candle: Dict[str, Any],
        indicator_state: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Return a signal dict or None.

        Signal format:
        {
            "direction": "LONG" | "SHORT",
            "entry_price": float,
            "stop_loss": float,
            "take_profit": float,
            "timestamp": datetime,
        }
        """


class BacktestStrategyAdapter(ABC):
    """Optional marker interface for backtest adapters."""

    @abstractmethod
    def on_bar(self, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Handle one bar and return a signal if generated."""
