---
description: Initialize the 43fx Forex trading system development environment
---

# Initialize Development Environment

// turbo-all

1. Create a Python virtual environment (if one doesn't already exist):
   ```bash
   python3 -m venv venv
   ```

2. Activate the venv and install dependencies:
   ```bash
   source venv/bin/activate && pip install -r requirements.txt
   ```

3. Copy the example config to create a local settings file (skip if `config/settings.yaml` already exists):
   ```bash
   cp -n config/settings_example.yaml config/settings.yaml
   ```

4. Create the `data/`, `results/`, and `logs/` directories if they don't exist:
   ```bash
   mkdir -p data results logs
   ```

5. Run the test suite to verify the environment:
   ```bash
   source venv/bin/activate && python -m unittest discover -s tests -p "test_*.py"
   ```

All 7 tests should pass. The environment is now ready for development.

## Quick Reference

| Action | Command |
|---|---|
| Run backtest | `source venv/bin/activate && python -m src.backtest.backtest_runner --config config/settings.yaml --csv data/EURUSD_H1.csv` |
| Run live (dummy) | `source venv/bin/activate && python -m src.live.live_runner --config config/settings.yaml` |
| Run tests | `source venv/bin/activate && python -m unittest discover -s tests -p "test_*.py"` |
