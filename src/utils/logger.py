"""Logging helpers for console and file output."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict


def setup_logger(logging_config: Dict[str, Any]) -> logging.Logger:
    """Configure root logger from config and return it."""
    level_name = str(logging_config.get("level", "INFO")).upper()
    log_file = logging_config.get("file", "logs/trading_system.log")

    level = getattr(logging, level_name, logging.INFO)

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("forex_system")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    return logger
