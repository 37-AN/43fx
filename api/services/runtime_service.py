from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.execution.dummy_broker import DummyBroker
from src.execution.oanda_client import OandaClient

from .config_service import ConfigService
from .state_manager import RuntimeStateManager


class RuntimeService:
    def __init__(self, config_service: ConfigService, state: RuntimeStateManager) -> None:
        self.config_service = config_service
        self.state = state

    def _build_broker(self, config: Dict[str, Any]):
        broker_cfg = config.get("broker", {})
        broker_type = str(broker_cfg.get("type", "dummy")).lower()
        if broker_type == "dummy":
            return DummyBroker(), broker_type
        if broker_type == "oanda":
            return (
                OandaClient(
                    api_base_url=str(broker_cfg.get("api_base_url", "")),
                    account_id=str(broker_cfg.get("account_id", "")),
                    api_key=str(broker_cfg.get("api_key", "")),
                ),
                broker_type,
            )
        return DummyBroker(), "dummy"

    def get_equity_points(self) -> List[Dict[str, Any]]:
        curve_path = Path("results/equity_curve.csv")
        if not curve_path.exists():
            return []

        df = pd.read_csv(curve_path)
        if df.empty or "equity" not in df.columns:
            return []

        df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
        df = df.dropna(subset=["equity"])
        if df.empty:
            return []

        rolling_max = df["equity"].cummax()
        drawdown = ((rolling_max - df["equity"]) / rolling_max).fillna(0.0)

        points = []
        for row, dd in zip(df.to_dict("records"), drawdown.tolist()):
            points.append(
                {
                    "timestamp": str(row.get("timestamp", "")),
                    "equity": float(row.get("equity", 0.0)),
                    "drawdown_pct": float(dd),
                }
            )
        return points

    def get_trade_history(self, limit: int) -> List[Dict[str, Any]]:
        trade_path = Path("results/trades.csv")
        if not trade_path.exists():
            return []

        df = pd.read_csv(trade_path)
        if df.empty:
            return []

        df = df.tail(limit)
        rows = []
        for row in df.to_dict("records"):
            rows.append(
                {
                    "direction": str(row.get("direction", "UNKNOWN")),
                    "entry_time": self._safe_str(row.get("entry_time")),
                    "exit_time": self._safe_str(row.get("exit_time")),
                    "pnl": self._safe_float(row.get("pnl_usd", row.get("pnl", 0.0))),
                    "r_multiple": self._safe_float(row.get("r_multiple", 0.0)),
                    "ai_probability": self._safe_optional_float(row.get("ai_probability")),
                    "size": self._safe_optional_float(row.get("lots")),
                    "entry": self._safe_optional_float(row.get("entry_price")),
                    "exit": self._safe_optional_float(row.get("exit_price")),
                }
            )
        return rows[::-1]

    def get_last_trade_time(self) -> Optional[str]:
        runtime_snapshot = self.state.snapshot()
        if runtime_snapshot.get("last_trade_time"):
            return runtime_snapshot["last_trade_time"]

        trade_path = Path("results/trades.csv")
        if not trade_path.exists():
            return None

        df = pd.read_csv(trade_path)
        if df.empty:
            return None

        last = df.iloc[-1]
        value = last.get("exit_time") or last.get("entry_time")
        return self._safe_str(value)

    def refresh_account_metrics(self) -> Dict[str, Any]:
        config = self.config_service.load_copy()

        broker_type = str(config.get("broker", {}).get("type", "dummy")).lower()
        open_positions = 0
        equity = 0.0

        try:
            if self.state.is_live_running() and self.state.live_runner is not None:
                broker = self.state.live_runner.broker
            else:
                broker, broker_type = self._build_broker(config)

            equity = float(broker.get_account_equity())
            positions = broker.get_open_positions()
            open_positions = len(positions)
        except Exception:
            snapshot = self.state.snapshot()
            equity = float(snapshot.get("current_equity", 0.0))

        metrics = self.state.update_equity_metrics(equity)
        return {
            "broker_type": broker_type,
            "open_positions": open_positions,
            **metrics,
        }

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _safe_optional_float(value: Any) -> Optional[float]:
        try:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value)
        return text if text and text.lower() != "nan" else None
