"""Live trading loop for polling candles and placing trades."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from threading import Event
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from src.config_loader import load_config
from src.ai.model_resolver import resolve_ai_model
from src.execution.broker_api_base import BrokerAPI
from src.execution.dummy_broker import DummyBroker
from src.execution.oanda_client import OandaClient
from src.risk.risk_engine import RiskEngine
from src.strategy.ema_trend_atr import EMATrendATRConfig, EMATrendATRStrategyLogic
from src.utils.logger import setup_logger


class LiveTradingRunner:
    """Orchestrates live signal generation and broker order placement."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger(config.get("logging", {}))

        self.broker = self._build_broker(config)
        self.risk_engine = self._build_risk_engine(config)
        self.strategy = self._build_strategy(config)

        trading_cfg = config.get("trading", {})
        self.instrument = trading_cfg.get("instruments", ["EURUSD"])[0]
        self.timeframe = trading_cfg.get("timeframe", "H1")
        self.polling_interval_seconds = int(trading_cfg.get("polling_interval_seconds", 20))

        self.session_cfg = config.get("session", {})

        self.last_processed_timestamp: Optional[pd.Timestamp] = None
        self.day_start_equity: Optional[float] = None
        self.current_day = None
        self.peak_equity = 0.0

    def _build_broker(self, config: Dict[str, Any]) -> BrokerAPI:
        broker_cfg = config.get("broker", {})
        broker_type = str(broker_cfg.get("type", "dummy")).lower()

        if broker_type == "dummy":
            return DummyBroker()

        if broker_type == "oanda":
            return OandaClient(
                api_base_url=str(broker_cfg.get("api_base_url", "")),
                account_id=str(broker_cfg.get("account_id", "")),
                api_key=str(broker_cfg.get("api_key", "")),
            )

        raise ValueError(f"Unsupported broker type: {broker_type}")

    def _build_risk_engine(self, config: Dict[str, Any]) -> RiskEngine:
        risk_cfg = config.get("risk", {})
        strategy_cfg = config.get("strategy", {})
        return RiskEngine(
            risk_per_trade=float(risk_cfg.get("risk_per_trade", 0.01)),
            max_daily_loss_pct=float(risk_cfg.get("max_daily_loss_pct", 0.03)),
            max_drawdown_pct=float(risk_cfg.get("max_drawdown_pct", 0.12)),
            pip_value_per_lot=float(risk_cfg.get("pip_value_per_lot", 10.0)),
            account_currency=str(risk_cfg.get("account_currency", "USD")),
            pip_size=float(strategy_cfg.get("pip_size", 0.0001)),
        )

    def _build_strategy(self, config: Dict[str, Any]) -> EMATrendATRStrategyLogic:
        strategy_cfg = config.get("strategy", {})
        ai_cfg = config.get("ai", {})
        
        ai_enabled = bool(ai_cfg.get("enabled", False))
        ai_model = None

        if ai_enabled:
            ai_model, provider, model_ref = resolve_ai_model({"ai": ai_cfg})
            if ai_model is None:
                self.logger.warning(
                    "AI filter is enabled but model provider=%s ref=%s could not be loaded. "
                    "Disabling AI filter for this live session.",
                    provider,
                    model_ref,
                )
                ai_enabled = False
        
        strategy_config = EMATrendATRConfig(
            ema_fast_period=int(strategy_cfg.get("ema_fast_period", 20)),
            ema_slow_period=int(strategy_cfg.get("ema_slow_period", 50)),
            atr_period=int(strategy_cfg.get("atr_period", 14)),
            atr_min_pips=float(strategy_cfg.get("atr_min_pips", 7.0)),
            pullback_lookback_min=int(strategy_cfg.get("pullback_lookback_min", 3)),
            pullback_lookback_max=int(strategy_cfg.get("pullback_lookback_max", 5)),
            swing_lookback=int(strategy_cfg.get("swing_lookback", 5)),
            pip_size=float(strategy_cfg.get("pip_size", 0.0001)),
            ai_enabled=ai_enabled,
        )
        return EMATrendATRStrategyLogic(
            strategy_config,
            ai_config=ai_cfg,
            ai_model=ai_model,
        )

    def _candles_to_df(self, candles: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(candles)
        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        for column in ["open", "high", "low", "close", "volume"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        return df.dropna()

    def _risk_state(self, equity: float) -> Dict[str, float]:
        today = datetime.now(tz=timezone.utc).date()
        if self.current_day != today:
            self.current_day = today
            self.day_start_equity = equity

        if self.day_start_equity in (None, 0):
            daily_loss_pct = 0.0
        else:
            daily_loss_pct = max((self.day_start_equity - equity) / self.day_start_equity, 0.0)

        self.peak_equity = max(self.peak_equity, equity)
        drawdown_pct = 0.0 if self.peak_equity == 0 else max((self.peak_equity - equity) / self.peak_equity, 0.0)

        return {"daily_loss_pct": daily_loss_pct, "drawdown_pct": drawdown_pct}

    def _has_position_for_direction(self, direction: str) -> bool:
        positions = self.broker.get_open_positions()
        return any(
            pos.get("instrument") == self.instrument and str(pos.get("direction", "")).upper() == direction
            for pos in positions
        )

    def run_forever(
        self,
        stop_event: Optional[Event] = None,
        on_order: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_heartbeat: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Start continuous polling loop."""
        self.logger.info("Live runner started for %s %s", self.instrument, self.timeframe)
        while True:
            if stop_event is not None and stop_event.is_set():
                self.logger.info("Live runner stop event received. Exiting loop.")
                break

            try:
                candles_raw = self.broker.get_latest_candles(self.instrument, self.timeframe, limit=500)
                candles_df = self._candles_to_df(candles_raw)
                if candles_df.empty:
                    self.logger.warning("No candles available; sleeping.")
                    if stop_event is not None:
                        stop_event.wait(self.polling_interval_seconds)
                    else:
                        time.sleep(self.polling_interval_seconds)
                    continue

                last_closed_ts = candles_df.index[-1]
                if self.last_processed_timestamp is not None and last_closed_ts <= self.last_processed_timestamp:
                    if stop_event is not None:
                        stop_event.wait(self.polling_interval_seconds)
                    else:
                        time.sleep(self.polling_interval_seconds)
                    continue

                self.last_processed_timestamp = last_closed_ts
                indicators_df = self.strategy.build_indicator_frame(candles_df)

                candle = {
                    "timestamp": last_closed_ts.to_pydatetime(),
                    "open": float(candles_df.iloc[-1]["open"]),
                    "high": float(candles_df.iloc[-1]["high"]),
                    "low": float(candles_df.iloc[-1]["low"]),
                    "close": float(candles_df.iloc[-1]["close"]),
                    "volume": float(candles_df.iloc[-1]["volume"]),
                }

                signal = self.strategy.generate_signal(
                    candle=candle,
                    indicator_state={"df": indicators_df},
                    context={"session": self.session_cfg},
                )

                if not signal:
                    self.logger.info("No signal on candle close %s", last_closed_ts.isoformat())
                    if stop_event is not None:
                        stop_event.wait(self.polling_interval_seconds)
                    else:
                        time.sleep(self.polling_interval_seconds)
                    continue

                direction = signal["direction"]
                if self._has_position_for_direction(direction):
                    self.logger.info("Skipped %s signal: existing %s position present.", direction, direction)
                    if stop_event is not None:
                        stop_event.wait(self.polling_interval_seconds)
                    else:
                        time.sleep(self.polling_interval_seconds)
                    continue

                equity = float(self.broker.get_account_equity())
                risk_state = self._risk_state(equity)
                if on_heartbeat is not None:
                    on_heartbeat(
                        {
                            "equity": equity,
                            "daily_loss_pct": risk_state["daily_loss_pct"],
                            "drawdown_pct": risk_state["drawdown_pct"],
                        }
                    )
                if not self.risk_engine.can_open_new_trade(
                    daily_realized_loss_pct=risk_state["daily_loss_pct"],
                    current_drawdown_pct=risk_state["drawdown_pct"],
                ):
                    self.logger.warning(
                        "Risk guardrail hit (daily_loss=%.2f%%, drawdown=%.2f%%)",
                        risk_state["daily_loss_pct"] * 100,
                        risk_state["drawdown_pct"] * 100,
                    )
                    if stop_event is not None:
                        stop_event.wait(self.polling_interval_seconds)
                    else:
                        time.sleep(self.polling_interval_seconds)
                    continue

                lots = self.risk_engine.size_position(
                    entry_price=float(signal["entry_price"]),
                    stop_loss_price=float(signal["stop_loss"]),
                    equity=equity,
                    symbol=self.instrument,
                )
                if lots <= 0:
                    self.logger.warning("Calculated lot size is zero; skipping order.")
                    if stop_event is not None:
                        stop_event.wait(self.polling_interval_seconds)
                    else:
                        time.sleep(self.polling_interval_seconds)
                    continue

                response = self.broker.place_market_order(
                    instrument=self.instrument,
                    direction=direction,
                    lots=lots,
                    stop_loss=float(signal["stop_loss"]),
                    take_profit=float(signal["take_profit"]),
                )
                self.logger.info("Order placed: %s", response)
                if on_order is not None:
                    on_order(response)

            except Exception as exc:  # noqa: BLE001
                self.logger.exception("Live loop error: %s", exc)

            if stop_event is not None:
                stop_event.wait(self.polling_interval_seconds)
            else:
                time.sleep(self.polling_interval_seconds)


def main() -> None:
    """CLI entrypoint for live runner."""
    parser = argparse.ArgumentParser(description="Run live trading loop.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    runner = LiveTradingRunner(config)
    runner.run_forever()


if __name__ == "__main__":
    main()
