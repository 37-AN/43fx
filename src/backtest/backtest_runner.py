"""Backtrader runner and performance report generation."""

from __future__ import annotations

import argparse
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import backtrader as bt
import pandas as pd

from src.ai.trade_filter_model import load_model as load_ai_model
from src.config_loader import load_config
from src.data.data_loader import load_historical_csv
from src.data.synthetic_generator import generate_trending_series
from src.risk.risk_engine import RiskEngine
from src.strategy.ema_trend_atr import EMATrendATRConfig, EMATrendATRStrategyLogic


class PandasDataFeed(bt.feeds.PandasData):
    """Custom Pandas feed mapping standardized OHLCV columns."""

    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
    )


class EMATrendATRBacktraderStrategy(bt.Strategy):
    """Backtrader wrapper using shared strategy logic and risk engine."""

    params = (
        ("strategy_config", None),
        ("session_config", None),
        ("risk_engine", None),
        ("instrument", "EURUSD"),
        ("ai_config", None),
        ("ai_model", None),
    )

    def __init__(self) -> None:
        self.strategy_logic = EMATrendATRStrategyLogic(
            self.params.strategy_config,
            ai_config=self.params.ai_config,
            ai_model=self.params.ai_model,
        )
        self.risk_engine: RiskEngine = self.params.risk_engine
        self.session_config: Dict[str, Any] = self.params.session_config or {}
        self.instrument: str = self.params.instrument

        self.order = None
        self.trades_log: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.current_day = None
        self.day_start_equity = None
        self.peak_equity = self.broker.getvalue()

        # For enriched trade logging: track the pending entry context
        self._pending_entry_ctx: Optional[Dict[str, Any]] = None

    def _snapshot_df(self, window: int = 300) -> pd.DataFrame:
        size = min(len(self.data), window)
        if size <= 0:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        rows = []
        for idx in range(-size + 1, 1):
            dt = bt.num2date(self.data.datetime[idx])
            rows.append(
                {
                    "datetime": pd.Timestamp(dt, tz="UTC"),
                    "open": float(self.data.open[idx]),
                    "high": float(self.data.high[idx]),
                    "low": float(self.data.low[idx]),
                    "close": float(self.data.close[idx]),
                    "volume": float(self.data.volume[idx]),
                }
            )

        df = pd.DataFrame(rows).set_index("datetime")
        return df

    def _compute_risk_state(self) -> Dict[str, float]:
        equity = float(self.broker.getvalue())
        dt = bt.num2date(self.data.datetime[0]).date()

        if self.current_day != dt:
            self.current_day = dt
            self.day_start_equity = equity

        if self.day_start_equity in (None, 0):
            daily_loss_pct = 0.0
        else:
            daily_loss_pct = max((self.day_start_equity - equity) / self.day_start_equity, 0.0)

        self.peak_equity = max(self.peak_equity, equity)
        drawdown_pct = 0.0 if self.peak_equity == 0 else max((self.peak_equity - equity) / self.peak_equity, 0.0)

        return {"daily_loss_pct": daily_loss_pct, "drawdown_pct": drawdown_pct}

    def next(self) -> None:
        equity = float(self.broker.getvalue())
        self.equity_curve.append(
            {
                "timestamp": bt.num2date(self.data.datetime[0]).isoformat(),
                "equity": equity,
            }
        )

        if self.order:
            return

        if self.position:
            return

        snapshot = self._snapshot_df()
        if snapshot.empty:
            return

        indicators_df = self.strategy_logic.build_indicator_frame(snapshot)
        bar_dt = bt.num2date(self.data.datetime[0])
        if bar_dt.tzinfo is None:
            bar_dt = bar_dt.replace(tzinfo=timezone.utc)
        candle = {
            "timestamp": bar_dt,
            "open": float(self.data.open[0]),
            "high": float(self.data.high[0]),
            "low": float(self.data.low[0]),
            "close": float(self.data.close[0]),
            "volume": float(self.data.volume[0]),
        }
        context = {"session": self.session_config}
        signal = self.strategy_logic.generate_signal(candle, {"df": indicators_df}, context)
        if not signal:
            return

        risk_state = self._compute_risk_state()
        if not self.risk_engine.can_open_new_trade(
            daily_realized_loss_pct=risk_state["daily_loss_pct"],
            current_drawdown_pct=risk_state["drawdown_pct"],
        ):
            return

        lots = self.risk_engine.size_position(
            entry_price=signal["entry_price"],
            stop_loss_price=signal["stop_loss"],
            equity=equity,
            symbol=self.instrument,
        )
        if lots <= 0:
            return

        size_units = max(int(lots * 100000), 1)
        if signal["direction"] == "LONG":
            self.order = self.buy_bracket(
                size=size_units,
                stopprice=signal["stop_loss"],
                limitprice=signal["take_profit"],
            )[0]
        else:
            self.order = self.sell_bracket(
                size=size_units,
                stopprice=signal["stop_loss"],
                limitprice=signal["take_profit"],
            )[0]

        # --- Capture indicator context at entry for enriched trade logging ---
        latest_ind = indicators_df.iloc[-1]
        prev_ind = indicators_df.iloc[-2] if len(indicators_df) >= 2 else latest_ind
        pip_size = self.strategy_logic.config.pip_size

        atr_val = float(latest_ind.get("atr", 0.0))
        atr_pips = atr_val / pip_size if pip_size > 0 else 0.0

        # ATR percentile over available history
        atr_series = indicators_df["atr"].dropna()
        if len(atr_series) > 1:
            atr_percentile = float((atr_series < atr_val).mean())
        else:
            atr_percentile = 0.5

        sl_pips = abs(signal["entry_price"] - signal["stop_loss"]) / pip_size if pip_size > 0 else 0.0
        risk_amount_usd = sl_pips * self.risk_engine.pip_value_per_lot * lots

        self._pending_entry_ctx = {
            "entry_time": candle["timestamp"].isoformat(),
            "direction": signal["direction"],
            "entry_price": signal["entry_price"],
            "stop_loss": signal["stop_loss"],
            "take_profit": signal["take_profit"],
            "lots": lots,
            "atr_pips": atr_pips,
            "ema_fast": float(latest_ind["ema_fast"]),
            "ema_slow": float(latest_ind["ema_slow"]),
            "ema_fast_prev": float(prev_ind["ema_fast"]),
            "ema_slow_prev": float(prev_ind["ema_slow"]),
            "recent_atr_percentile": atr_percentile,
            "entry_hour": bar_dt.hour,
            "pip_size": pip_size,
            "risk_amount_usd": risk_amount_usd,
        }

    def notify_trade(self, trade: bt.Trade) -> None:
        """Called when a trade is opened, updated, or closed."""
        if not trade.isclosed:
            return

        ctx = self._pending_entry_ctx or {}
        risk_amount_usd = float(ctx.get("risk_amount_usd", 0.0))
        pnl = float(trade.pnlcomm)

        if risk_amount_usd > 0:
            r_multiple = pnl / risk_amount_usd
        else:
            r_multiple = 0.0

        exit_dt = bt.num2date(self.data.datetime[0])
        if exit_dt.tzinfo is None:
            exit_dt = exit_dt.replace(tzinfo=timezone.utc)

        enriched_row = {
            **ctx,
            "exit_time": exit_dt.isoformat(),
            "exit_price": float(self.data.close[0]),
            "pnl_usd": pnl,
            "r_multiple": round(r_multiple, 4),
        }
        self.trades_log.append(enriched_row)
        self._pending_entry_ctx = None

    def notify_order(self, order: bt.Order) -> None:
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None


def _compute_stats(strategy: EMATrendATRBacktraderStrategy, starting_equity: float) -> Dict[str, float]:
    ending_equity = float(strategy.broker.getvalue())
    total_return = (ending_equity - starting_equity) / starting_equity if starting_equity else 0.0

    win_count = 0
    loss_count = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for row in strategy.trades_log:
        pnl = float(row.get("pnl_usd", 0.0))
        if pnl > 0:
            win_count += 1
            gross_profit += pnl
        elif pnl < 0:
            loss_count += 1
            gross_loss += abs(pnl)

    total_closed = win_count + loss_count
    win_rate = (win_count / total_closed) if total_closed else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    equity_series = pd.Series([point["equity"] for point in strategy.equity_curve], dtype="float64")
    rolling_max = equity_series.cummax()
    dd = ((rolling_max - equity_series) / rolling_max).fillna(0.0)
    max_drawdown = float(dd.max()) if not dd.empty else 0.0

    return {
        "starting_equity": starting_equity,
        "ending_equity": ending_equity,
        "total_return": total_return,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
    }


def _print_diagnostics(strategy: EMATrendATRBacktraderStrategy) -> None:
    """Print diagnostic filter counters when zero trades were triggered."""
    diag = strategy.strategy_logic.diagnostics.summary()
    print("\n" + "=" * 50)
    print("STRATEGY DIAGNOSTICS (0 trades triggered)")
    print("=" * 50)
    print(f"  Bars evaluated:       {diag['bars_evaluated']}")
    print(f"  Session filter pass:  {diag['session_pass_count']}")
    print(f"  Session filter fail:  {diag['session_fail_count']}")
    print(f"  ATR filter pass:      {diag['atr_pass_count']}")
    print(f"  ATR filter fail:      {diag['atr_fail_count']}")
    print(f"  Trend LONG pass:      {diag['trend_long_pass_count']}")
    print(f"  Trend SHORT pass:     {diag['trend_short_pass_count']}")
    print(f"  Trend neutral:        {diag['trend_neutral_count']}")
    print(f"  Pullback LONG pass:   {diag['pullback_long_pass_count']}")
    print(f"  Pullback LONG fail:   {diag['pullback_long_fail_count']}")
    print(f"  Pullback SHORT pass:  {diag['pullback_short_pass_count']}")
    print(f"  Pullback SHORT fail:  {diag['pullback_short_fail_count']}")
    print(f"  Reclaim LONG pass:    {diag['reclaim_long_pass_count']}")
    print(f"  Reclaim LONG fail:    {diag['reclaim_long_fail_count']}")
    print(f"  Reject SHORT pass:    {diag['reject_short_pass_count']}")
    print(f"  Reject SHORT fail:    {diag['reject_short_fail_count']}")
    print(f"  Signals generated:    {diag['signal_count']}")
    print("=" * 50)


def _load_data(config: Dict[str, Any], csv_path: str) -> pd.DataFrame:
    """Load price data from CSV or generate synthetic trending data from config."""
    data_cfg = config.get("data", {})
    use_synthetic = data_cfg.get("use_synthetic", False)

    if use_synthetic:
        bars = int(data_cfg.get("bars", 3000))
        drift = float(data_cfg.get("drift_per_bar", 0.00005))
        vol = float(data_cfg.get("volatility", 0.0003))
        seed = int(data_cfg.get("seed", 42))
        print(f"[DATA] Using synthetic trending series: bars={bars}, drift={drift}, vol={vol}")
        return generate_trending_series(
            bars=bars,
            drift_per_bar=drift,
            volatility=vol,
            seed=seed,
        )

    return load_historical_csv(csv_path)


def _resolve_ai_model(config: Dict[str, Any]) -> Any:
    """Load the AI filter model if ai.enabled is True, else return None."""
    ai_config = config.get("ai", {})
    if not ai_config.get("enabled", False):
        return None

    model_path = str(ai_config.get("model_path", "models/trade_filter.pkl"))
    model = load_ai_model(model_path)
    if model is None:
        print(f"[AI] WARNING: ai.enabled=True but model not found at {model_path}. AI filter disabled.")
    else:
        print(f"[AI] Model loaded from {model_path}")
    return model


def run_backtest(config: Dict[str, Any], csv_path: str) -> Dict[str, float]:
    """Run backtest and persist results under results/."""
    data = _load_data(config, csv_path)

    strategy_cfg = config.get("strategy", {})
    risk_cfg = config.get("risk", {})
    session_cfg = config.get("session", {})
    ai_config = config.get("ai", {})
    instruments = config.get("trading", {}).get("instruments", ["EURUSD"])
    instrument = instruments[0]

    ai_enabled = bool(ai_config.get("enabled", False))
    ai_model = _resolve_ai_model(config) if ai_enabled else None
    if ai_enabled and ai_model is None:
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
        debug=bool(strategy_cfg.get("debug", False)),
        relaxed_mode=bool(strategy_cfg.get("relaxed_mode", False)),
        ai_enabled=ai_enabled,
    )

    risk_engine = RiskEngine(
        risk_per_trade=float(risk_cfg.get("risk_per_trade", 0.01)),
        max_daily_loss_pct=float(risk_cfg.get("max_daily_loss_pct", 0.03)),
        max_drawdown_pct=float(risk_cfg.get("max_drawdown_pct", 0.12)),
        pip_value_per_lot=float(risk_cfg.get("pip_value_per_lot", 10.0)),
        account_currency=str(risk_cfg.get("account_currency", "USD")),
        pip_size=float(strategy_cfg.get("pip_size", 0.0001)),
    )

    cerebro = bt.Cerebro()
    starting_equity = 10_000.0
    cerebro.broker.setcash(starting_equity)

    feed = PandasDataFeed(dataname=data)
    cerebro.adddata(feed)

    cerebro.addstrategy(
        EMATrendATRBacktraderStrategy,
        strategy_config=strategy_config,
        session_config=session_cfg,
        risk_engine=risk_engine,
        instrument=instrument,
        ai_config=ai_config,
        ai_model=ai_model,
    )

    strategies = cerebro.run()
    strat = strategies[0]

    stats = _compute_stats(strat, starting_equity=starting_equity)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    pd.DataFrame(strat.trades_log).to_csv(results_dir / "trades.csv", index=False)
    pd.DataFrame(strat.equity_curve).to_csv(results_dir / "equity_curve.csv", index=False)

    # --- Metrics summary ---
    trade_count = len(strat.trades_log)
    print(f"\nTotal Trades: {trade_count}")
    print(f"Total Return: {stats['total_return']:.2%}")
    print(f"Win Rate: {stats['win_rate']:.2%}")
    print(f"Max Drawdown: {stats['max_drawdown']:.2%}")
    print(f"Profit Factor: {stats['profit_factor']:.2f}")
    print(f"Start Equity: {stats['starting_equity']:.2f}")
    print(f"End Equity: {stats['ending_equity']:.2f}")

    if trade_count == 0:
        _print_diagnostics(strat)

    return stats


def main() -> None:
    """CLI entrypoint for backtest execution."""
    parser = argparse.ArgumentParser(description="Run Forex strategy backtest.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--csv", default="", help="Path to historical OHLCV CSV (ignored when data.use_synthetic=true).")
    args = parser.parse_args()

    config = load_config(args.config)
    run_backtest(config, args.csv)


if __name__ == "__main__":
    main()
