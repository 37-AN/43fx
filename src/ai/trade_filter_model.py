"""AI trade filter model: feature engineering, training, and persistence."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover - fallback path
    joblib = None

logger = logging.getLogger("forex_system")

LONDON_START_UTC = 7
LONDON_END_UTC = 16
NY_START_UTC = 12
NY_END_UTC = 21


@dataclass
class TradeFeatureRow:
    """Container for a single training row."""

    features: List[float] | np.ndarray
    label: int


class _FallbackClassifier:
    """Lightweight fallback classifier used only when sklearn is unavailable."""

    def __init__(self) -> None:
        self.threshold = 0.0
        self.pos_mean = 1.0
        self.neg_mean = -1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_FallbackClassifier":
        score = X[:, 0] if X.ndim == 2 and X.shape[1] > 0 else np.zeros(len(y))
        pos = score[y == 1]
        neg = score[y == 0]
        self.pos_mean = float(np.mean(pos)) if len(pos) else 1.0
        self.neg_mean = float(np.mean(neg)) if len(neg) else -1.0
        self.threshold = (self.pos_mean + self.neg_mean) / 2.0
        return self

    def predict_proba(self, X):
        probs = []
        for row in X:
            score = float(row[0]) if len(row) else 0.0
            distance = self.pos_mean - self.neg_mean
            scale = abs(distance) if abs(distance) > 1e-9 else 1.0
            p = 1.0 / (1.0 + np.exp(-(score - self.threshold) / scale))
            probs.append([1.0 - p, p])
        return np.array(probs, dtype=float)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


def _resolve_ai_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return ai-config whether caller passed full config or ai-only config."""
    if "ai" in config and isinstance(config.get("ai"), dict):
        return config["ai"]
    return config


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_entry_hour(trade_row: Dict[str, Any]) -> int:
    """Extract UTC entry hour from row fields."""
    if trade_row.get("entry_hour") is not None:
        return _safe_int(trade_row.get("entry_hour"), 12)

    entry_time = trade_row.get("entry_time")
    if entry_time is None:
        return 12

    ts = pd.to_datetime(entry_time, utc=True, errors="coerce")
    if pd.isna(ts):
        return 12
    return int(ts.hour)


def build_feature_vector(trade_row: dict, config: dict) -> list[float]:
    """Convert a trade row into a numeric feature vector."""
    ai_cfg = _resolve_ai_config(config)
    feature_cfg = ai_cfg.get("features", {})

    use_session_flags = bool(feature_cfg.get("use_session_flags", True))
    use_volatility_features = bool(feature_cfg.get("use_volatility_features", True))
    use_trend_slope = bool(feature_cfg.get("use_trend_slope", True))

    pip_size = _safe_float(trade_row.get("pip_size"), 0.0001)
    if pip_size <= 0:
        pip_size = 0.0001

    entry_price = _safe_float(trade_row.get("entry_price"), 0.0)
    ema20 = _safe_float(trade_row.get("ema_fast"), entry_price)
    ema50 = _safe_float(trade_row.get("ema_slow"), entry_price)
    ema20_prev = _safe_float(trade_row.get("ema_fast_prev"), ema20)
    ema50_prev = _safe_float(trade_row.get("ema_slow_prev"), ema50)

    atr_pips_at_entry = _safe_float(trade_row.get("atr_pips"), 0.0)
    ema20_slope = (ema20 - ema20_prev) / pip_size if use_trend_slope else 0.0
    ema50_slope = (ema50 - ema50_prev) / pip_size if use_trend_slope else 0.0
    distance_price_to_ema20 = (entry_price - ema20) / pip_size
    distance_price_to_ema50 = (entry_price - ema50) / pip_size
    recent_trend_strength = abs(ema20 - ema50) / pip_size
    recent_atr_percentile = (
        _safe_float(trade_row.get("recent_atr_percentile"), 0.5)
        if use_volatility_features
        else 0.5
    )

    if use_session_flags:
        entry_hour = _extract_entry_hour(trade_row)
        session_is_london = 1.0 if LONDON_START_UTC <= entry_hour < LONDON_END_UTC else 0.0
        session_is_ny = 1.0 if NY_START_UTC <= entry_hour < NY_END_UTC else 0.0
    else:
        session_is_london = 0.0
        session_is_ny = 0.0

    direction = str(trade_row.get("direction", "LONG")).upper()
    direction_long = 1.0 if direction == "LONG" else 0.0
    direction_short = 1.0 if direction == "SHORT" else 0.0

    return [
        atr_pips_at_entry,
        ema20_slope,
        ema50_slope,
        distance_price_to_ema20,
        distance_price_to_ema50,
        recent_trend_strength,
        recent_atr_percentile,
        session_is_london,
        session_is_ny,
        direction_long,
        direction_short,
    ]


def build_feature_vector_from_signal(
    signal: Dict[str, Any],
    indicators_df: pd.DataFrame,
    config: Dict[str, Any],
    pip_size: float = 0.0001,
) -> list[float]:
    """Build feature vector for a live/backtest candidate signal."""
    latest = indicators_df.iloc[-1]
    prev = indicators_df.iloc[-2] if len(indicators_df) >= 2 else latest

    atr_value = _safe_float(latest.get("atr"), 0.0)
    atr_pips = atr_value / pip_size if pip_size > 0 else 0.0

    atr_series = indicators_df["atr"].dropna() if "atr" in indicators_df.columns else pd.Series(dtype=float)
    recent_atr_percentile = float((atr_series < atr_value).mean()) if len(atr_series) > 1 else 0.5

    timestamp = signal.get("timestamp")
    entry_hour = int(getattr(timestamp, "hour", 12))

    row = {
        "entry_price": signal.get("entry_price"),
        "direction": signal.get("direction"),
        "atr_pips": atr_pips,
        "ema_fast": _safe_float(latest.get("ema_fast"), _safe_float(signal.get("entry_price"))),
        "ema_slow": _safe_float(latest.get("ema_slow"), _safe_float(signal.get("entry_price"))),
        "ema_fast_prev": _safe_float(prev.get("ema_fast"), _safe_float(latest.get("ema_fast"))),
        "ema_slow_prev": _safe_float(prev.get("ema_slow"), _safe_float(latest.get("ema_slow"))),
        "recent_atr_percentile": recent_atr_percentile,
        "entry_hour": entry_hour,
        "pip_size": pip_size,
    }
    return build_feature_vector(row, config)


def extract_training_dataset(trades_csv_path: str, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load trades CSV, build features, and return (X, y)."""
    df = pd.read_csv(trades_csv_path)
    if df.empty:
        return np.empty((0, 11), dtype=float), np.empty((0,), dtype=int)

    rows: List[TradeFeatureRow] = []
    for _, row in df.iterrows():
        trade = row.to_dict()
        features = build_feature_vector(trade, config)
        r_multiple = _safe_float(trade.get("r_multiple"), 0.0)
        label = 1 if r_multiple >= 1.5 else 0
        rows.append(TradeFeatureRow(features=features, label=label))

    X = np.array([r.features for r in rows], dtype=float)
    y = np.array([r.label for r in rows], dtype=int)
    return X, y


def train_model(X: np.ndarray, y: np.ndarray, config: dict):
    """Train scikit-learn classifier based on ai.model_type."""
    ai_cfg = _resolve_ai_config(config)
    model_type = str(ai_cfg.get("model_type", "logistic")).lower()

    if model_type == "random_forest":
        try:
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                random_state=42,
                class_weight="balanced",
            )
        except ModuleNotFoundError:  # pragma: no cover - environment fallback
            model = _FallbackClassifier()
    else:
        try:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(
                max_iter=500,
                random_state=42,
                class_weight="balanced",
            )
        except ModuleNotFoundError:  # pragma: no cover - environment fallback
            model = _FallbackClassifier()

    model.fit(X, y)
    return model


def save_model(model, path: str) -> None:
    """Save model to disk via joblib."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if joblib is not None:
        joblib.dump(model, model_path)
    else:
        with model_path.open("wb") as handle:
            pickle.dump(model, handle)
    logger.info("Saved AI model to %s", model_path)


def load_model(path: str):
    """Load model from disk if present, else return None."""
    model_path = Path(path)
    if not model_path.exists():
        return None
    if joblib is not None:
        return joblib.load(model_path)
    with model_path.open("rb") as handle:
        return pickle.load(handle)
