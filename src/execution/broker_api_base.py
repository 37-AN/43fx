"""Broker API abstraction for live execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BrokerAPI(ABC):
    """Abstract broker interface used by live runner."""

    @abstractmethod
    def get_account_equity(self) -> float:
        """Return current account equity."""

    @abstractmethod
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Return open positions."""

    @abstractmethod
    def place_market_order(
        self,
        instrument: str,
        direction: str,
        lots: float,
        stop_loss: float,
        take_profit: float,
    ) -> Dict[str, Any]:
        """Place a market order and return broker response payload."""

    @abstractmethod
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Return open/pending orders."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel order by identifier."""

    @abstractmethod
    def get_latest_candles(self, instrument: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
        """Return latest candle list with OHLCV and timestamp fields."""
