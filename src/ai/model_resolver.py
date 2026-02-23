"""Resolve AI model provider and construct model objects."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from src.ai.ollama_trade_filter import OllamaTradeFilterModel
from src.ai.trade_filter_model import load_model as load_local_model


def resolve_ai_model(config: Dict[str, Any]) -> Tuple[Any, str, str]:
    """Return (model_instance_or_none, provider, model_ref)."""
    ai_cfg = config.get("ai", config)

    provider = str(ai_cfg.get("provider", "local")).lower()
    model_path = str(ai_cfg.get("model_path", "models/trade_filter.pkl"))

    # Backward-compatible autodetection from model_path.
    if model_path.startswith("ollama:"):
        provider = "ollama"

    if provider == "ollama":
        configured_name = str(ai_cfg.get("ollama_model", "")).strip() or str(os.getenv("OLLAMA_MODEL", "")).strip()
        model_name = configured_name or model_path.replace("ollama:", "", 1).strip()
        if not model_name:
            model_name = "llama3.2"

        base_url = str(ai_cfg.get("ollama_base_url", "")).strip() or str(
            os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        )
        timeout_seconds = int(ai_cfg.get("ollama_timeout_seconds", 20))
        model = OllamaTradeFilterModel(
            model_name=model_name,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        return (model if model.health_check() else None, "ollama", model_name)

    return load_local_model(model_path), "local", model_path
