from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigService:
    def __init__(self, config_path: str = "config/settings.yaml") -> None:
        self.config_path = Path(config_path)

    def load(self) -> Dict[str, Any]:
        with self.config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def save(self, config: Dict[str, Any]) -> None:
        with self.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    def load_copy(self) -> Dict[str, Any]:
        return deepcopy(self.load())

    def safe_update(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        config = self.load()
        applied: Dict[str, Any] = {}

        if payload.get("risk_per_trade") is not None:
            value = float(payload["risk_per_trade"])
            config.setdefault("risk", {})["risk_per_trade"] = value
            applied["risk_per_trade"] = value

        if payload.get("max_daily_loss_pct") is not None:
            value = float(payload["max_daily_loss_pct"])
            config.setdefault("risk", {})["max_daily_loss_pct"] = value
            applied["max_daily_loss_pct"] = value

        if payload.get("max_drawdown_pct") is not None:
            value = float(payload["max_drawdown_pct"])
            config.setdefault("risk", {})["max_drawdown_pct"] = value
            applied["max_drawdown_pct"] = value

        if payload.get("ai_probability_threshold") is not None:
            value = float(payload["ai_probability_threshold"])
            config.setdefault("ai", {})["probability_threshold"] = value
            applied["ai.probability_threshold"] = value

        self.save(config)
        return applied
