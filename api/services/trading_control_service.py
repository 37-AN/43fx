from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict

from src.live.live_runner import LiveTradingRunner

from .config_service import ConfigService
from .logging_service import setup_api_logger
from .state_manager import RuntimeStateManager


class TradingControlService:
    def __init__(self, config_service: ConfigService, state: RuntimeStateManager) -> None:
        self.config_service = config_service
        self.state = state
        self.logger = setup_api_logger()

    def start(self) -> Dict[str, str]:
        if self.state.is_live_running():
            return {"status": "ok", "detail": "Live strategy already running"}

        config = self.config_service.load_copy()
        stop_event = threading.Event()
        runner = LiveTradingRunner(config)

        def on_order(order: Dict[str, Any]) -> None:
            self.state.mark_trade_time(order.get("timestamp", datetime.now(timezone.utc).isoformat()))

        def on_heartbeat(payload: Dict[str, Any]) -> None:
            equity = float(payload.get("equity", 0.0))
            self.state.update_equity_metrics(equity)

        def target() -> None:
            try:
                self.logger.info("starting_live_loop", extra={"event": "strategy.start"})
                runner.run_forever(stop_event=stop_event, on_order=on_order, on_heartbeat=on_heartbeat)
            finally:
                self.logger.info("stopped_live_loop", extra={"event": "strategy.stop"})
                self.state.clear_live_runtime()

        thread = threading.Thread(target=target, name="live-trading-loop", daemon=True)
        self.state.set_live_runtime(runner=runner, stop_event=stop_event, thread=thread)
        thread.start()
        return {"status": "ok", "detail": "Live strategy started"}

    def stop(self) -> Dict[str, str]:
        if not self.state.is_live_running() or self.state.live_stop_event is None:
            self.state.clear_live_runtime()
            return {"status": "ok", "detail": "Live strategy already stopped"}

        self.state.live_stop_event.set()
        thread = self.state.live_thread
        if thread is not None:
            thread.join(timeout=3)

        self.state.clear_live_runtime()
        return {"status": "ok", "detail": "Live strategy stopped"}

    def restart(self) -> Dict[str, str]:
        self.stop()
        return self.start()

    def apply_runtime_risk_updates(self, applied: Dict[str, Any]) -> None:
        runner = self.state.live_runner
        if runner is None:
            return

        risk_engine = runner.risk_engine
        if "risk_per_trade" in applied:
            risk_engine.risk_per_trade = float(applied["risk_per_trade"])
        if "max_daily_loss_pct" in applied:
            risk_engine.max_daily_loss_pct = float(applied["max_daily_loss_pct"])
        if "max_drawdown_pct" in applied:
            risk_engine.max_drawdown_pct = float(applied["max_drawdown_pct"])

        if "ai.probability_threshold" in applied:
            runner.strategy.ai_config["probability_threshold"] = float(applied["ai.probability_threshold"])
