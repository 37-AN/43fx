from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class RuntimeStateManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.started_at = datetime.now(timezone.utc)

        self.live_thread: Optional[threading.Thread] = None
        self.live_stop_event: Optional[threading.Event] = None
        self.live_runner: Any = None

        self.last_trade_time: Optional[str] = None
        self.last_backtest_summary: Dict[str, float] = {}
        self.last_ai_train_summary: Dict[str, Any] = {
            "samples": 0,
            "positives": 0,
            "negatives": 0,
            "training_accuracy": 0.0,
        }

        self.current_equity: float = 0.0
        self.peak_equity: float = 0.0
        self.current_drawdown_pct: float = 0.0
        self.daily_loss_pct: float = 0.0
        self._current_day: Optional[str] = None
        self._day_start_equity: Optional[float] = None

    def set_live_runtime(self, runner: Any, stop_event: threading.Event, thread: threading.Thread) -> None:
        with self._lock:
            self.live_runner = runner
            self.live_stop_event = stop_event
            self.live_thread = thread

    def clear_live_runtime(self) -> None:
        with self._lock:
            self.live_runner = None
            self.live_stop_event = None
            self.live_thread = None

    def is_live_running(self) -> bool:
        with self._lock:
            return bool(self.live_thread and self.live_thread.is_alive())

    def mark_trade_time(self, value: str) -> None:
        with self._lock:
            self.last_trade_time = value

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "current_equity": self.current_equity,
                "peak_equity": self.peak_equity,
                "current_drawdown_pct": self.current_drawdown_pct,
                "daily_loss_pct": self.daily_loss_pct,
                "last_trade_time": self.last_trade_time,
                "system_uptime": str(datetime.now(timezone.utc) - self.started_at).split(".")[0],
            }

    def update_equity_metrics(self, equity: float) -> Dict[str, float]:
        with self._lock:
            today = datetime.now(timezone.utc).date().isoformat()
            if self._current_day != today:
                self._current_day = today
                self._day_start_equity = equity

            if self._day_start_equity in (None, 0):
                self.daily_loss_pct = 0.0
            else:
                self.daily_loss_pct = max((self._day_start_equity - equity) / self._day_start_equity, 0.0)

            self.current_equity = equity
            self.peak_equity = max(self.peak_equity, equity)
            self.current_drawdown_pct = (
                0.0 if self.peak_equity <= 0 else max((self.peak_equity - equity) / self.peak_equity, 0.0)
            )
            return {
                "equity": self.current_equity,
                "peak_equity": self.peak_equity,
                "daily_loss_pct": self.daily_loss_pct,
                "current_drawdown_pct": self.current_drawdown_pct,
            }

    def set_backtest_summary(self, summary: Dict[str, float]) -> None:
        with self._lock:
            self.last_backtest_summary = dict(summary)

    def set_ai_training_summary(self, summary: Dict[str, Any]) -> None:
        with self._lock:
            self.last_ai_train_summary = dict(summary)
