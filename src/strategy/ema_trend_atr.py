"""EMA trend + ATR pullback strategy implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import backtrader as bt
import pandas as pd

from src.strategy.base_strategy import BaseSignalStrategy
from src.utils.time_utils import is_within_session

logger = logging.getLogger("forex_system")


def compute_EMA(series: pd.Series, period: int) -> pd.Series:
    """Compute exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_ATR(df: pd.DataFrame, period: int) -> pd.Series:
    """Compute Average True Range from OHLC dataframe."""
    prev_close = df["close"].shift(1)
    tr_components = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()


@dataclass
class EMATrendATRConfig:
    """Config container for EMA trend ATR strategy."""

    ema_fast_period: int = 20
    ema_slow_period: int = 50
    atr_period: int = 14
    atr_min_pips: float = 7.0
    pullback_lookback_min: int = 3
    pullback_lookback_max: int = 5
    swing_lookback: int = 5
    pip_size: float = 0.0001

    # --- diagnostic / validation extensions ---
    debug: bool = False
    relaxed_mode: bool = False
    ai_enabled: bool = False


@dataclass
class DiagnosticCounters:
    """Accumulates per-bar filter pass/fail counts for post-run diagnostics."""

    bars_evaluated: int = 0
    session_pass: int = 0
    session_fail: int = 0
    atr_pass: int = 0
    atr_fail: int = 0
    trend_long_pass: int = 0
    trend_short_pass: int = 0
    trend_neutral: int = 0
    pullback_long_pass: int = 0
    pullback_long_fail: int = 0
    pullback_short_pass: int = 0
    pullback_short_fail: int = 0
    reclaim_long_pass: int = 0
    reclaim_long_fail: int = 0
    reject_short_pass: int = 0
    reject_short_fail: int = 0
    signal_count: int = 0

    def summary(self) -> Dict[str, int]:
        """Return a flat dictionary of all counters."""
        return {
            "bars_evaluated": self.bars_evaluated,
            "session_pass_count": self.session_pass,
            "session_fail_count": self.session_fail,
            "atr_pass_count": self.atr_pass,
            "atr_fail_count": self.atr_fail,
            "trend_long_pass_count": self.trend_long_pass,
            "trend_short_pass_count": self.trend_short_pass,
            "trend_neutral_count": self.trend_neutral,
            "pullback_long_pass_count": self.pullback_long_pass,
            "pullback_long_fail_count": self.pullback_long_fail,
            "pullback_short_pass_count": self.pullback_short_pass,
            "pullback_short_fail_count": self.pullback_short_fail,
            "reclaim_long_pass_count": self.reclaim_long_pass,
            "reclaim_long_fail_count": self.reclaim_long_fail,
            "reject_short_pass_count": self.reject_short_pass,
            "reject_short_fail_count": self.reject_short_fail,
            "signal_count": self.signal_count,
        }


class EMATrendATRStrategyLogic(BaseSignalStrategy):
    """Pure-Python strategy logic reusable by backtest and live modules."""

    def __init__(
        self,
        config: EMATrendATRConfig,
        ai_config: Optional[Dict[str, Any]] = None,
        ai_model: Optional[Any] = None,
    ):
        self.config = config
        self.diagnostics = DiagnosticCounters()

        # --- AI trade filter (optional) ---
        self.ai_config: Dict[str, Any] = ai_config or {}
        self.ai_model = ai_model
        self._ai_model_loaded = ai_model is not None
        self._ai_active = (
            config.ai_enabled
            and ai_model is not None
            and self.ai_config.get("enabled", False)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _effective_atr_min_pips(self) -> float:
        """Return ATR threshold, halved when relaxed_mode is active."""
        base = self.config.atr_min_pips
        return base * 0.5 if self.config.relaxed_mode else base

    def _effective_pullback_lookback(self) -> int:
        """Return pullback lookback, reduced to 2 when relaxed_mode is active."""
        return 2 if self.config.relaxed_mode else self.config.pullback_lookback_max

    def _session_enabled(self) -> bool:
        """Return False when relaxed_mode disables the session filter."""
        return not self.config.relaxed_mode

    def _has_pullback_touch(
        self,
        df: pd.DataFrame,
        ema_col: str,
        direction: str,
    ) -> bool:
        lookback = self._effective_pullback_lookback()
        recent = df.iloc[-(lookback + 1) : -1]
        if recent.empty:
            return False

        if direction == "LONG":
            touched = (recent["low"] <= recent[ema_col]).any()
        else:
            touched = (recent["high"] >= recent[ema_col]).any()

        return bool(touched)

    def _build_signal(
        self,
        direction: str,
        entry_price: float,
        atr_value: float,
        swing_low: float,
        swing_high: float,
        timestamp: datetime,
    ) -> Dict[str, Any]:
        if direction == "LONG":
            sl = max(swing_low, entry_price - 1.5 * atr_value)
            risk = entry_price - sl
            tp = entry_price + 2.0 * risk
        else:
            sl = min(swing_high, entry_price + 1.5 * atr_value)
            risk = sl - entry_price
            tp = entry_price - 2.0 * risk

        return {
            "direction": direction,
            "entry_price": float(entry_price),
            "stop_loss": float(sl),
            "take_profit": float(tp),
            "timestamp": timestamp,
        }

    def _debug(self, msg: str) -> None:
        """Emit a debug-level log line when debug mode is on."""
        if self.config.debug:
            logger.debug("[STRATEGY_DIAG] %s", msg)

    def _apply_ai_filter(
        self,
        signal: Dict[str, Any],
        indicators_df: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        """Run the AI trade filter on a candidate signal.

        Returns the signal unchanged if AI is disabled or approves the trade,
        or ``None`` if the model rejects it.

        # TODO: Plug in more advanced AI/ML models here in the future
        #       (e.g., neural networks, reinforcement-learning agents).
        """
        if self.config.ai_enabled and not self._ai_model_loaded and self.ai_config.get("enabled", False):
            model_path = str(self.ai_config.get("model_path", "models/trade_filter.pkl"))
            from src.ai.trade_filter_model import load_model as load_ai_model

            self.ai_model = load_ai_model(model_path)
            self._ai_model_loaded = True
            self._ai_active = self.ai_model is not None
            if not self._ai_active:
                logger.warning("AI enabled but model missing at %s. Passing signal through.", model_path)

        if not self._ai_active or self.ai_model is None:
            return signal

        from src.ai.trade_filter_model import build_feature_vector_from_signal

        features = build_feature_vector_from_signal(
            signal=signal,
            indicators_df=indicators_df,
            config=self.ai_config,
            pip_size=self.config.pip_size,
        )

        threshold = float(self.ai_config.get("probability_threshold", 0.55))

        try:
            proba = self.ai_model.predict_proba([features])[0]
            p_positive = float(proba[1]) if len(proba) > 1 else float(proba[0])
        except Exception as exc:
            logger.warning("AI predict_proba failed (%s); passing signal through.", exc)
            return signal

        if p_positive >= threshold:
            self._debug(
                f"AI APPROVED: p={p_positive:.3f} >= threshold={threshold:.3f}"
            )
            logger.info(
                "AI filter APPROVED %s trade (p=%.3f >= %.3f)",
                signal["direction"], p_positive, threshold,
            )
            return signal

        self._debug(
            f"AI REJECTED: p={p_positive:.3f} < threshold={threshold:.3f}"
        )
        logger.info(
            "AI filter REJECTED %s trade (p=%.3f < %.3f)",
            signal["direction"], p_positive, threshold,
        )
        return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        candle: Dict[str, Any],
        indicator_state: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Generate LONG/SHORT signal from latest indicator state and context."""
        df = indicator_state["df"]
        min_bars = max(self.config.ema_slow_period, self.config.atr_period) + 2
        if len(df) < min_bars:
            self._debug(f"Not enough bars ({len(df)} < {min_bars})")
            return None

        timestamp = candle.get("timestamp")
        if not isinstance(timestamp, datetime):
            self._debug("Missing or invalid timestamp")
            return None

        self.diagnostics.bars_evaluated += 1

        # ---- Session filter ----
        session_cfg = context.get("session", {})
        if self._session_enabled():
            if not is_within_session(timestamp, session_cfg):
                self.diagnostics.session_fail += 1
                self._debug(f"Session REJECTED at {timestamp}")
                return None
            self.diagnostics.session_pass += 1
            self._debug(f"Session PASSED at {timestamp}")
        else:
            # relaxed mode: auto-pass session
            self.diagnostics.session_pass += 1
            self._debug(f"Session BYPASSED (relaxed_mode) at {timestamp}")

        latest = df.iloc[-1]
        ema_fast = latest["ema_fast"]
        ema_slow = latest["ema_slow"]
        atr_value = latest["atr"]

        if pd.isna(ema_fast) or pd.isna(ema_slow) or pd.isna(atr_value):
            self._debug("Indicator NaN — skipping")
            return None

        # ---- ATR filter ----
        atr_pips = atr_value / self.config.pip_size
        atr_threshold = self._effective_atr_min_pips()
        if atr_pips < atr_threshold:
            self.diagnostics.atr_fail += 1
            self._debug(
                f"ATR REJECTED: {atr_pips:.1f} pips < {atr_threshold:.1f} threshold"
            )
            return None
        self.diagnostics.atr_pass += 1
        self._debug(f"ATR PASSED: {atr_pips:.1f} pips >= {atr_threshold:.1f}")

        close_price = float(latest["close"])
        swing_slice = df.iloc[-self.config.swing_lookback :]
        swing_low = float(swing_slice["low"].min())
        swing_high = float(swing_slice["high"].max())

        # ---- Trend + pullback + reclaim (LONG) ----
        if ema_fast > ema_slow:
            self.diagnostics.trend_long_pass += 1
            self._debug(
                f"Trend LONG: ema_fast={ema_fast:.5f} > ema_slow={ema_slow:.5f}"
            )

            pullback = self._has_pullback_touch(df, "ema_fast", "LONG")
            if pullback:
                self.diagnostics.pullback_long_pass += 1
                self._debug("Pullback LONG PASSED")
            else:
                self.diagnostics.pullback_long_fail += 1
                self._debug("Pullback LONG FAILED — no touch within lookback")

            close_reclaim = close_price > float(latest["ema_fast"])
            if close_reclaim:
                self.diagnostics.reclaim_long_pass += 1
                self._debug(
                    f"Reclaim LONG PASSED: close={close_price:.5f} > ema_fast={float(latest['ema_fast']):.5f}"
                )
            else:
                self.diagnostics.reclaim_long_fail += 1
                self._debug(
                    f"Reclaim LONG FAILED: close={close_price:.5f} <= ema_fast={float(latest['ema_fast']):.5f}"
                )

            if pullback and close_reclaim:
                self.diagnostics.signal_count += 1
                self._debug(">>> LONG SIGNAL GENERATED <<<")
                raw_signal = self._build_signal(
                    direction="LONG",
                    entry_price=close_price,
                    atr_value=float(atr_value),
                    swing_low=swing_low,
                    swing_high=swing_high,
                    timestamp=timestamp,
                )
                return self._apply_ai_filter(raw_signal, df)

        # ---- Trend + pullback + reject (SHORT) ----
        if ema_fast < ema_slow:
            self.diagnostics.trend_short_pass += 1
            self._debug(
                f"Trend SHORT: ema_fast={ema_fast:.5f} < ema_slow={ema_slow:.5f}"
            )

            pullback = self._has_pullback_touch(df, "ema_fast", "SHORT")
            if pullback:
                self.diagnostics.pullback_short_pass += 1
                self._debug("Pullback SHORT PASSED")
            else:
                self.diagnostics.pullback_short_fail += 1
                self._debug("Pullback SHORT FAILED — no touch within lookback")

            close_reject = close_price < float(latest["ema_fast"])
            if close_reject:
                self.diagnostics.reject_short_pass += 1
                self._debug(
                    f"Reject SHORT PASSED: close={close_price:.5f} < ema_fast={float(latest['ema_fast']):.5f}"
                )
            else:
                self.diagnostics.reject_short_fail += 1
                self._debug(
                    f"Reject SHORT FAILED: close={close_price:.5f} >= ema_fast={float(latest['ema_fast']):.5f}"
                )

            if pullback and close_reject:
                self.diagnostics.signal_count += 1
                self._debug(">>> SHORT SIGNAL GENERATED <<<")
                raw_signal = self._build_signal(
                    direction="SHORT",
                    entry_price=close_price,
                    atr_value=float(atr_value),
                    swing_low=swing_low,
                    swing_high=swing_high,
                    timestamp=timestamp,
                )
                return self._apply_ai_filter(raw_signal, df)

        # ---- No trend alignment ----
        if not (ema_fast > ema_slow) and not (ema_fast < ema_slow):
            self.diagnostics.trend_neutral += 1
            self._debug(
                f"Trend NEUTRAL: ema_fast={ema_fast:.5f} == ema_slow={ema_slow:.5f}"
            )

        return None

    def build_indicator_frame(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Create indicator-enriched dataframe from OHLCV candles."""
        df = candles.copy()
        df["ema_fast"] = compute_EMA(df["close"], self.config.ema_fast_period)
        df["ema_slow"] = compute_EMA(df["close"], self.config.ema_slow_period)
        df["atr"] = compute_ATR(df, self.config.atr_period)
        return df


class BacktraderEMATrendATR(bt.Strategy):
    """Backtrader-native indicator setup for this strategy family."""

    params = (
        ("ema_fast_period", 20),
        ("ema_slow_period", 50),
        ("atr_period", 14),
    )

    def __init__(self) -> None:
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.params.ema_fast_period)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.params.ema_slow_period)
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
