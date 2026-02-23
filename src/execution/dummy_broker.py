"""In-memory dummy broker implementation for live simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from src.execution.broker_api_base import BrokerAPI


@dataclass
class DummyBroker(BrokerAPI):
    """Minimal in-memory broker for local testing without real execution."""

    starting_equity: float = 10_000.0
    _equity: float = field(default=10_000.0, init=False)
    _positions: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _orders: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _candles: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._equity = self.starting_equity

    def seed_candles(self, instrument: str, candles: List[Dict[str, Any]]) -> None:
        """Seed candle cache for integration tests and dry-runs."""
        self._candles[instrument] = candles

    def get_account_equity(self) -> float:
        return float(self._equity)

    def get_open_positions(self) -> List[Dict[str, Any]]:
        return list(self._positions)

    def place_market_order(
        self,
        instrument: str,
        direction: str,
        lots: float,
        stop_loss: float,
        take_profit: float,
    ) -> Dict[str, Any]:
        order = {
            "id": f"order_{len(self._orders) + 1}",
            "instrument": instrument,
            "direction": direction,
            "lots": lots,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "status": "FILLED",
        }
        self._orders.append(order)
        self._positions.append(order.copy())
        return order

    def get_open_orders(self) -> List[Dict[str, Any]]:
        return [order for order in self._orders if order.get("status") == "OPEN"]

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        for order in self._orders:
            if order["id"] == order_id:
                order["status"] = "CANCELLED"
                return {"status": "cancelled", "order_id": order_id}
        return {"status": "not_found", "order_id": order_id}

    def get_latest_candles(self, instrument: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
        candles = self._candles.get(instrument, [])
        return candles[-limit:]
