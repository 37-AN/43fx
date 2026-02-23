"""CLI training entrypoint for the trade filter model."""

from __future__ import annotations

import argparse

import numpy as np

from src.ai.trade_filter_model import extract_training_dataset, save_model, train_model
from src.config_loader import load_config
from src.utils.logger import setup_logger


def main() -> None:
    """Train and persist trade filter model from trades CSV."""
    parser = argparse.ArgumentParser(description="Train AI trade filter model.")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to YAML config.")
    parser.add_argument("--trades-csv", default="results/trades.csv", help="Path to trades CSV.")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger(config.get("logging", {}))

    ai_cfg = config.get("ai", {})
    X, y = extract_training_dataset(args.trades_csv, config)

    samples = int(len(y))
    min_trades = int(ai_cfg.get("train_min_trades", 200))
    if samples < min_trades:
        logger.info("Skipping training: samples=%d below ai.train_min_trades=%d", samples, min_trades)
        return

    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))

    model = train_model(X, y, config)
    train_accuracy = float(np.mean(model.predict(X) == y)) if samples > 0 else 0.0

    model_path = str(ai_cfg.get("model_path", "models/trade_filter.pkl"))
    save_model(model, model_path)

    logger.info("samples=%d", samples)
    logger.info("class_balance: label_1=%d label_0=%d", positives, negatives)
    logger.info("train_accuracy=%.4f", train_accuracy)


if __name__ == "__main__":
    main()
