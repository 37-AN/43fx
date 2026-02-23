from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.ai.model_resolver import resolve_ai_model
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
        provider = str(ai_cfg.get("provider", "local")).lower()
        if provider == "ollama":
            summary = {
                "status": "ok",
                "trained": False,
                "samples": 0,
                "positives": 0,
                "negatives": 0,
                "training_accuracy": 0.0,
                "model_path": str(ai_cfg.get("ollama_model", "")) or str(ai_cfg.get("model_path", "ollama:llama3.2")),
            }
            self.state.set_ai_training_summary(summary)
            return summary

        try:
            X, y = extract_training_dataset(trades_csv, config)
        except FileNotFoundError:
            summary = {
                "status": "ok",
                "trained": False,
                "samples": 0,
                "positives": 0,
                "negatives": 0,
                "training_accuracy": 0.0,
                "model_path": str(ai_cfg.get("model_path", "models/trade_filter.pkl")),
            }
            self.state.set_ai_training_summary(summary)
            return summary
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

        model_loaded, provider, model_ref = self._availability(config)

        return {
            "enabled": bool(enabled) and model_loaded if enabled else False,
            "model_loaded": model_loaded,
            "provider": provider,
            "model_path": model_ref,
        }

    def status(self) -> Dict[str, Any]:
        config = self.config_service.load_copy()
        ai_cfg = config.get("ai", {})
        model_loaded, provider, model_ref = self._availability(config)

        training = self.state.last_ai_train_summary
        return {
            "enabled": bool(ai_cfg.get("enabled", False)) and model_loaded,
            "model_loaded": model_loaded,
            "provider": provider,
            "model_path": model_ref,
            "probability_threshold": float(ai_cfg.get("probability_threshold", 0.55)),
            "training_sample_size": int(training.get("samples", 0)),
            "class_balance": {
                "positive": int(training.get("positives", 0)),
                "negative": int(training.get("negatives", 0)),
            },
        }

    @staticmethod
    def _availability(config: Dict[str, Any]) -> tuple[bool, str, str]:
        model, provider, model_ref = resolve_ai_model(config)
        if provider == "local" and model is None:
            # keep backward compatibility with direct path checks for local model files
            ai_cfg = config.get("ai", {})
            model_path = str(ai_cfg.get("model_path", "models/trade_filter.pkl"))
            model = load_model(model_path)
            model_ref = model_path
        return (model is not None, provider, model_ref)
