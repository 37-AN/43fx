"""Ollama-backed trade filter model compatible with predict_proba/predict."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import requests


@dataclass
class OllamaTradeFilterModel:
    """Small adapter exposing sklearn-like methods over Ollama HTTP API."""

    model_name: str
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: int = 20

    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=(1, min(self.timeout_seconds, 2)))
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models", [])
            names = [str(row.get("name", "")) for row in models]
            return any(name == self.model_name or name.startswith(f"{self.model_name}:") for name in names)
        except Exception:
            return False

    def predict_proba(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        probabilities = []
        for features in X:
            p = self._infer_probability(features)
            probabilities.append([1.0 - p, p])
        return np.array(probabilities, dtype=float)

    def predict(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= 0.5).astype(int)

    def _infer_probability(self, features: Sequence[float]) -> float:
        prompt = self._build_prompt(features)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=(2, self.timeout_seconds),
            )
            response.raise_for_status()
            result = response.json()
            raw_text = str(result.get("response", "{}")).strip()
            data = json.loads(raw_text) if raw_text else {}
            probability = float(data.get("probability", 0.5))
        except Exception:
            probability = 0.5

        if probability < 0.0:
            return 0.0
        if probability > 1.0:
            return 1.0
        return probability

    @staticmethod
    def _build_prompt(features: Sequence[float]) -> str:
        features_list = [float(x) for x in features]
        return (
            "You are a quantitative forex trade quality classifier. "
            "Return ONLY strict JSON with one key: probability. "
            "probability must be a float from 0.0 to 1.0 indicating chance this setup is high quality. "
            f"Features: {features_list}."
        )
