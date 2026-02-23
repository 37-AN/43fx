"""Oanda client skeleton implementing the broker interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import requests

from src.execution.broker_api_base import BrokerAPI


@dataclass
class OandaClient(BrokerAPI):
    """Placeholder Oanda implementation.

    This class intentionally keeps method bodies as skeletons showing where
    authenticated HTTP requests should be implemented.
    """

    api_base_url: str
    account_id: str
    api_key: str

    @property
    def _headers(self) -> Dict[str, str]:
        """Build request headers for Oanda endpoints."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_account_equity(self) -> float:
        """Fetch current account NAV/equity from Oanda account summary endpoint."""
        # Example endpoint:
        # GET /v3/accounts/{accountID}/summary
        # response = requests.get(url, headers=self._headers, timeout=15)
        # return float(response.json()["account"]["NAV"])
        raise NotImplementedError("Implement Oanda account equity request.")

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Fetch current open positions."""
        # Example endpoint:
        # GET /v3/accounts/{accountID}/openPositions
        raise NotImplementedError("Implement Oanda open positions request.")

    def place_market_order(
        self,
        instrument: str,
        direction: str,
        lots: float,
        stop_loss: float,
        take_profit: float,
    ) -> Dict[str, Any]:
        """Submit market order with attached SL/TP."""
        # Example endpoint:
        # POST /v3/accounts/{accountID}/orders
        # units = int(lots * 100000) with sign based on direction
        # payload includes stopLossOnFill and takeProfitOnFill.
        raise NotImplementedError("Implement Oanda market order placement.")

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Fetch current pending orders."""
        # Example endpoint:
        # GET /v3/accounts/{accountID}/pendingOrders
        raise NotImplementedError("Implement Oanda open orders request.")

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel order by ID."""
        # Example endpoint:
        # PUT /v3/accounts/{accountID}/orders/{orderSpecifier}/cancel
        raise NotImplementedError("Implement Oanda order cancellation.")

    def get_latest_candles(self, instrument: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch latest candles for an instrument/timeframe."""
        # Example endpoint:
        # GET /v3/instruments/{instrument}/candles?granularity=H1&count={limit}
        # Return normalized list of dicts:
        # [{"timestamp": ..., "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}, ...]
        raise NotImplementedError("Implement Oanda candle request.")
