from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.backtest.backtest_runner import run_backtest

from .config_service import ConfigService
from .state_manager import RuntimeStateManager


class BacktestService:
    def __init__(self, config_service: ConfigService, state: RuntimeStateManager) -> None:
        self.config_service = config_service
        self.state = state

    def run(
        self,
        instrument: str,
        start_date: date,
        end_date: date,
        use_synthetic: bool,
    ) -> Dict[str, float]:
        config = self.config_service.load_copy()
        config.setdefault("trading", {})["instruments"] = [instrument]
        config.setdefault("data", {})["use_synthetic"] = bool(use_synthetic)

        csv_path = ""
        if not use_synthetic:
            source_path = self._resolve_csv(instrument)
            if source_path is None:
                raise FileNotFoundError(f"No CSV found for instrument {instrument}")
            csv_path = self._filter_csv(source_path, start_date, end_date, instrument)

        summary = run_backtest(config, csv_path)
        self.state.set_backtest_summary(summary)
        return {k: float(v) for k, v in summary.items()}

    @staticmethod
    def _resolve_csv(instrument: str) -> str | None:
        data_dir = Path("data")
        if not data_dir.exists():
            return None

        sanitized = instrument.replace("/", "").replace("-", "").upper()
        candidates = sorted(data_dir.glob("*.csv"))
        for file in candidates:
            name = file.stem.upper().replace("-", "").replace("_", "")
            if sanitized in name:
                return str(file)

        for file in candidates:
            if "EURUSD" in instrument.upper() and "EURUSD" in file.stem.upper().replace("-", ""):
                return str(file)
        return str(candidates[0]) if candidates else None

    @staticmethod
    def _filter_csv(path: str, start_date: date, end_date: date, instrument: str) -> str:
        df = pd.read_csv(path)
        if df.empty:
            return path

        dt_col = None
        for candidate in ["datetime", "timestamp", "time", "date"]:
            if candidate in df.columns:
                dt_col = candidate
                break

        if dt_col is None:
            return path

        series = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
        start_ts = pd.Timestamp(start_date).tz_localize("UTC")
        end_ts = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered = df[(series >= start_ts) & (series <= end_ts)].copy()

        if filtered.empty:
            filtered = df.copy()

        tmp_dir = Path("/tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_path = tmp_dir / f"backtest_{instrument.replace('/', '')}_{start_date}_{end_date}.csv"
        filtered.to_csv(out_path, index=False)
        return str(out_path)
