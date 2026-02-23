from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.ai.trade_filter_model import extract_training_dataset, load_model, save_model, train_model

from .config_service import ConfigService
from .state_manager import RuntimeStateManager


class AIService:
    def __init__(self, config_service: ConfigService, state: RuntimeStateManager) -> None:
        self.config_service = config_service
        self.state = state

    def train(self, trades_csv: str = "results/trades.csv") -> Dict[str, Any]:
        config = self.config_service.load_copy()
        ai_cfg = config.get("ai", {})

        X, y = extract_training_dataset(trades_csv, config)
        samples = int(len(y))
        positives = int(np.sum(y == 1)) if samples else 0
        negatives = int(np.sum(y == 0)) if samples else 0

        min_trades = int(ai_cfg.get("train_min_trades", 200))
        model_path = str(ai_cfg.get("model_path", "models/trade_filter.pkl"))

        if samples < min_trades:
            summary = {
                "status": "ok",
                "trained": False,
                "samples": samples,
                "positives": positives,
                "negatives": negatives,
                "training_accuracy": 0.0,
                "model_path": model_path,
            }
            self.state.set_ai_training_summary(summary)
            return summary

        model = train_model(X, y, config)
        training_accuracy = float(np.mean(model.predict(X) == y)) if samples else 0.0
        save_model(model, model_path)

        summary = {
            "status": "ok",
            "trained": True,
            "samples": samples,
            "positives": positives,
            "negatives": negatives,
            "training_accuracy": training_accuracy,
            "model_path": model_path,
        }
        self.state.set_ai_training_summary(summary)
        return summary

    def set_enabled(self, enabled: bool) -> Dict[str, Any]:
        config = self.config_service.load_copy()
        config.setdefault("ai", {})["enabled"] = bool(enabled)
        self.config_service.save(config)

        model_path = str(config.get("ai", {}).get("model_path", "models/trade_filter.pkl"))
        model_loaded = load_model(model_path) is not None

        return {
            "enabled": bool(enabled) and model_loaded if enabled else False,
            "model_loaded": model_loaded,
            "model_path": model_path,
        }

    def status(self) -> Dict[str, Any]:
        config = self.config_service.load_copy()
        ai_cfg = config.get("ai", {})
        model_path = str(ai_cfg.get("model_path", "models/trade_filter.pkl"))
        model_loaded = Path(model_path).exists() and load_model(model_path) is not None

        training = self.state.last_ai_train_summary
        return {
            "enabled": bool(ai_cfg.get("enabled", False)) and model_loaded,
            "model_loaded": model_loaded,
            "model_path": model_path,
            "probability_threshold": float(ai_cfg.get("probability_threshold", 0.55)),
            "training_sample_size": int(training.get("samples", 0)),
            "class_balance": {
                "positive": int(training.get("positives", 0)),
                "negative": int(training.get("negatives", 0)),
            },
        }
