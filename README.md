# Forex Trend-Following Trading System

Python architecture for a modular algorithmic Forex trading stack with shared strategy and risk logic for backtesting and live execution.

## Features

- Data layer for historical CSV and live candle polling
- EMA20/EMA50 + ATR trend-following strategy
- Reusable risk engine (percent-of-equity sizing)
- Broker abstraction with dummy and Oanda client skeleton
- Backtesting pipeline built on Backtrader
- Live runner loop with logging and error handling

## Project Layout

- `config/settings_example.yaml`: Example runtime configuration
- `src/`: Source code for data, strategy, risk, execution, backtest, live, and utilities
- `tests/`: Unit tests for risk sizing and strategy signaling
- `results/`: Backtest output CSV files

## Run Backtest

1. Copy and edit config:
   - `cp config/settings_example.yaml config/settings.yaml`
2. Put historical CSVs under `data/` (example: `data/EURUSD_H1.csv`)
3. Execute:
   - `python -m src.backtest.backtest_runner --config config/settings.yaml --csv data/EURUSD_H1.csv`

## Run Live (Dummy Broker)

1. Copy and edit config:
   - `cp config/settings_example.yaml config/settings.yaml`
2. Set broker type to `dummy`
3. Execute:
   - `python -m src.live.live_runner --config config/settings.yaml`

## Tests

- `python -m unittest discover -s tests -p "test_*.py"`
